"""
Atomic scaling module. Used for predicting extensive properties.
"""

import logging
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from torch_scatter import scatter_mean

DATA_INDEX = {"total_energy": 0, "forces": 2, "per_atom_energy": 1, "per_species_energy": 0}

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Kernel, Hyperparameter


def solver(X, y, regressor: Optional[str] = "NormalizedGaussianProcess", **kwargs):
    if regressor == "GaussianProcess":
        return gp(X, y, **kwargs)
    elif regressor == "NormalizedGaussianProcess":
        return normalized_gp(X, y, **kwargs)
    else:
        raise NotImplementedError(f"{regressor} is not implemented")


def normalized_gp(X, y, **kwargs):
    feature_rms = 1.0 / np.sqrt(np.average(X**2, axis=0))
    feature_rms = np.nan_to_num(feature_rms, 1)
    y_mean = torch.sum(y) / torch.sum(X)
    mean, std = base_gp(
        X,
        y - (torch.sum(X, axis=1) * y_mean).reshape(y.shape),
        NormalizedDotProduct,
        {"diagonal_elements": feature_rms},
        **kwargs,
    )
    return mean + y_mean, std


def gp(X, y, **kwargs):
    return base_gp(
        X, y, DotProduct, {"sigma_0": 0, "sigma_0_bounds": "fixed"}, **kwargs
    )


def base_gp(
    X,
    y,
    kernel,
    kernel_kwargs,
    alpha: Optional[float] = 0.1,
    max_iteration: int = 20,
    stride: Optional[int] = 1,
):

    if len(y.shape) == 1:
        y = y.reshape([-1, 1])

    if stride is not None:
        X = X[::stride]
        y = y[::stride]

    not_fit = True
    iteration = 0
    mean = None
    std = None
    while not_fit:
        print(f"GP fitting iteration {iteration} {alpha}")
        try:
            _kernel = kernel(**kernel_kwargs)
            gpr = GaussianProcessRegressor(kernel=_kernel, random_state=0, alpha=alpha)
            gpr = gpr.fit(X, y)

            vec = torch.diag(torch.ones(X.shape[1]))
            mean, std = gpr.predict(vec, return_std=True)

            mean = torch.as_tensor(mean, dtype=torch.get_default_dtype()).reshape([-1])
            # ignore all the off-diagonal terms
            std = torch.as_tensor(std, dtype=torch.get_default_dtype()).reshape([-1])
            likelihood = gpr.log_marginal_likelihood()

            res = torch.sqrt(
                torch.square(torch.matmul(X, mean.reshape([-1, 1])) - y).mean()
            )

            print(
                f"GP fitting: alpha {alpha}:\n"
                f"            residue {res}\n"
                f"            mean {mean} std {std}\n"
                f"            log marginal likelihood {likelihood}"
            )
            not_fit = False

        except Exception as e:
            print(f"GP fitting failed for alpha={alpha} and {e.args}")
            if alpha == 0 or alpha is None:
                logging.info("try a non-zero alpha")
                not_fit = False
                raise ValueError(
                    f"Please set the {alpha} to non-zero value. \n"
                    "The dataset energy is rank deficient to be solved with GP"
                )
            else:
                alpha = alpha * 2
                iteration += 1
                logging.debug(f"           increase alpha to {alpha}")

            if iteration >= max_iteration or not_fit is False:
                raise ValueError(
                    "Please set the per species shift and scale to zeros and ones. \n"
                    "The dataset energy is to diverge to be solved with GP"
                )

    return mean, std


class NormalizedDotProduct(Kernel):
    r"""Dot-Product kernel.
    .. math::
        k(x_i, x_j) = x_i \cdot A \cdot x_j
    """

    def __init__(self, diagonal_elements):
        # TODO: check shape
        self.diagonal_elements = diagonal_elements
        self.A = np.diag(diagonal_elements)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            K = (X.dot(self.A)).dot(X.T)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            K = (X.dot(self.A)).dot(Y.T)

        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).
        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return np.einsum("ij,ij,jj->i", X, X, self.A)

    def __repr__(self):
        return ""

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    @property
    def hyperparameter_diagonal_elements(self):
        return Hyperparameter("diagonal_elements", "numeric", "fixed")

def solver(X, y, regressor: Optional[str] = "NormalizedGaussianProcess", **kwargs):
    if regressor == "GaussianProcess":
        return gp(X, y, **kwargs)
    elif regressor == "NormalizedGaussianProcess":
        return normalized_gp(X, y, **kwargs)
    else:
        raise NotImplementedError(f"{regressor} is not implemented")

class AtomScaling(nn.Module):
    """
    Atomic extensive property rescaling module
    """
    def __init__(
            self,
            atoms: list[Atoms] = None,
            total_energy: list[float] = None,
            forces: list[np.ndarray] = None,
            atomic_numbers: list[np.ndarray] = None,
            num_atoms: list[float] = None,
            max_z: int = 94,
            scale_key: str = None,
            shift_key: str = None,
            init_scale: Union[torch.Tensor, float] = None,
            init_shift: Union[torch.Tensor, float] = None,
            trainable_scale: bool = False,
            trainable_shift: bool = False,
            verbose: bool = False,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            **kwargs
    ):
        """
        Args:
            forces: a list of atomic forces (np.ndarray) in each graph
            max_z: (int) maximum atomic number
                - if scale_key or shift_key is specified, max_z should be equal to the maximum atomic_number.
            scale_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms (default)
            shift_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean : default option is gaussian regression (NequIP)
                - per_species_energy_mean_linear_reg : an alternative choice is linear regression (M3GNet)
            init_scale (torch.Tensor or float)
            init_shift (torch.Tensor or float)
        """
        super().__init__()

        self.max_z = max_z
        self.device = device

        # === Data preprocessing ===
        if scale_key or shift_key:
            total_energy = torch.from_numpy(np.array(total_energy))
            forces = torch.from_numpy(np.concatenate(forces, axis=0)) if forces is not None else None
            if atomic_numbers is None:
                atomic_numbers = [atom.get_atomic_numbers() for atom in atoms]
            atomic_numbers = torch.from_numpy(np.concatenate(atomic_numbers, axis=0)).squeeze(-1).long() # (num_atoms,)
            # assert max_z == atomic_numbers.max().item(), "max_z should be equal to the maximum atomic_number"
            if num_atoms is None:
                num_atoms = [atom.positions.shape[0] for atom in atoms] # (N_GRAPHS, )
            num_atoms = torch.from_numpy(np.array(num_atoms))
            per_atom_energy = total_energy / num_atoms
            data_list = [total_energy, per_atom_energy, forces]

            assert num_atoms.size()[0] == total_energy.size()[0], "num_atoms and total_energy should have the same size, but got {} and {}".format(num_atoms.size()[0], total_energy.size()[0])
            if forces is not None:
                assert forces.size()[0] == atomic_numbers.size()[0], "forces and atomic_numbers should have the same length, but got {} and {}".format(forces.size()[0], atomic_numbers.size()[0])

            # === Calculate the scaling factors ===
            if scale_key == "per_species_energy_std" and shift_key == "per_species_energy_mean" \
                    and init_shift is None and init_scale is None:
                # Using gaussian regression two times to get the shift and scale is potentially unstable
                init_shift, init_scale = self.get_gaussian_statistics(atomic_numbers, num_atoms, total_energy)
            else:
                if shift_key and init_shift is None:
                    init_shift = self.get_statistics(shift_key, max_z, data_list, atomic_numbers, num_atoms)
                if scale_key and init_scale is None:
                    init_scale = self.get_statistics(scale_key, max_z, data_list, atomic_numbers, num_atoms)

        # === initial values are given ===
        if init_scale is None:
            init_scale = torch.ones(max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = torch.tensor(init_scale).repeat(max_z + 1)
        else:
            assert init_scale.size()[0] == max_z + 1

        if init_shift is None:
            init_shift = torch.zeros(max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = torch.tensor(init_shift).repeat(max_z + 1)
        else:
            assert init_shift.size()[0] == max_z + 1

        init_shift = init_shift.float()
        init_scale = init_scale.float()
        if trainable_scale is True:
            self.scale = torch.nn.Parameter(init_scale)
        else:
            self.register_buffer("scale", init_scale)

        if trainable_shift is True:
            self.shift = torch.nn.Parameter(init_shift)
        else:
            self.register_buffer("shift", init_shift)

        if verbose is True:
            print("Current scale: ", init_scale)
            print("Current shift: ", init_shift)

        self.to(device)

    def transform(self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Take the origin values from model and get the transformed values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        normalized_energies = curr_scale * atomic_energies + curr_shift
        return normalized_energies

    def inverse_transform(self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Take the transformed values and get the original values
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        unnormalized_energies = (atomic_energies - curr_shift) / curr_scale
        return unnormalized_energies

    def forward(self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Atomic_energies and atomic_numbers should have the same size
        """
        return self.transform(atomic_energies, atomic_numbers)

    def get_statistics(self, key, max_z, data_list, atomic_numbers, num_atoms) -> torch.Tensor:
        """
        Valid key:
            scale_key: valid options are:
                - total_energy_mean
                - per_atom_energy_mean
                - per_species_energy_mean
                - per_species_energy_mean_linear_reg : an alternative choice is linear regression
            shift_key: valid options are:
                - total_energy_std
                - per_atom_energy_std
                - per_species_energy_std
                - forces_rms
                - per_species_forces_rms
        """
        data = None
        for data_key in DATA_INDEX:
            if data_key in key:
                data = data_list[DATA_INDEX[data_key]]
        assert data is not None

        statistics = None
        if "mean" in key:
            if "per_species" in key:
                n_atoms = torch.repeat_interleave(repeats=num_atoms)
                if "linear_reg" in key:
                    features = bincount(atomic_numbers, n_atoms, minlength=self.max_z+1).numpy()
                    # print(features[0], features.shape)
                    data = data.numpy()
                    assert features.ndim == 2  # [batch, n_type]
                    features = features[(features > 0).any(axis=1)]  # deal with non-contiguous batch indexes
                    statistics = np.linalg.pinv(features.T.dot(features)).dot(features.T.dot(data))
                    statistics = torch.from_numpy(statistics)
                else:
                    N = bincount(atomic_numbers, num_atoms, minlength=self.max_z+1)
                    assert N.ndim == 2  # [batch, n_type]
                    N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
                    N = N.type(torch.get_default_dtype())
                    statistics, _ = solver(N, data, regressor="NormalizedGaussianProcess")
            else:
                statistics = torch.mean(data).item()
        elif "std" in key:
            if "per_species" in key:
                print("Warning: calculating per_species_energy_std for full periodic table systems is risky, please use per_species_forces_rms instead")
                n_atoms = torch.repeat_interleave(repeats=num_atoms)
                N = bincount(atomic_numbers, n_atoms, minlength=self.max_z+1)
                assert N.ndim == 2  # [batch, n_type]
                N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
                N = N.type(torch.get_default_dtype())
                _, statistics = solver(N, data, regressor="NormalizedGaussianProcess")
            else:
                statistics = torch.std(data).item()
        elif "rms" in key:
            if "per_species" in key:
                square = scatter_mean(data.square(), atomic_numbers, dim=0, dim_size=max_z+1)
                statistics = square.mean(axis=-1)
            else:
                statistics = torch.sqrt(torch.mean(data.square())).item()

        if isinstance(statistics, torch.Tensor) is not True:
            statistics = torch.tensor(statistics).repeat(max_z + 1)

        assert statistics.size()[0] == max_z + 1

        return statistics

    def get_gaussian_statistics(self, atomic_numbers: torch.Tensor, num_atoms: torch.Tensor, total_energy: torch.Tensor):
        """
        Get the gaussian process mean and variance
        """
        n_atoms = torch.repeat_interleave(repeats=num_atoms)
        N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
        assert N.ndim == 2  # [batch, n_type]
        N = N[(N > 0).any(dim=1)]  # deal with non-contiguous batch indexes
        N = N.type(torch.get_default_dtype())
        mean, std = solver(N, total_energy, regressor="NormalizedGaussianProcess")
        assert mean.size()[0] == self.max_z + 1
        assert std.size()[0] == self.max_z + 1
        return mean, std

def bincount(
    input: torch.Tensor, batch: Optional[torch.Tensor] = None, minlength: int = 0
):
    assert input.ndim == 1
    if batch is None:
        return torch.bincount(input, minlength=minlength)
    else:
        assert batch.shape == input.shape

        length = input.max().item() + 1
        if minlength == 0:
            minlength = length
        if length > minlength:
            raise ValueError(
                f"minlength {minlength} too small for input with integers up to and including {length}"
            )

        # Flatten indexes
        # Make each "class" in input into a per-input class.
        input_ = input + batch * minlength

        num_batch = batch.max() + 1

        return torch.bincount(input_, minlength=minlength * num_batch).reshape(
            num_batch, minlength
        )