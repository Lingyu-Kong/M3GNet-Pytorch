import torch
import numpy as np
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pymatgen.optimization.neighbors import find_points_in_spheres
from tqdm import tqdm

def get_fixed_radius_bonding(
    atoms: Atoms,
    cutoff: float,
    numerical_tol: float = 1e-8,
    pbc: bool = True,
):
    pbc_ = np.array(atoms.pbc, dtype=int)
    if np.all(pbc_ < 0.1) or pbc == False:
        lattice_matrix = np.array(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
            dtype=float,
        )
        pbc_ = np.array([0, 0, 0], dtype=int)
    else:
        lattice_matrix = np.ascontiguousarray(atoms.cell[:], dtype=float)

    cart_coords = np.ascontiguousarray(np.array(atoms.positions), dtype=float)
    r = float(cutoff)

    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=r,
        pbc=pbc_,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )
    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    images = images.astype(np.int64)
    distances = distances.astype(float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    return (
        center_indices[exclude_self],
        neighbor_indices[exclude_self],
        images[exclude_self],
        distances[exclude_self],
    )
    
def compute_threebody_indices(
    num_atoms: int,
    edge_index: np.ndarray, ## [2, num_edge]
    edge_length: np.ndarray,
    threebody_cutoff: float,
):
    """
    This function's implementation largely follows the matgl repo: https://github.com/materialsvirtuallab/matgl/tree/main
    """
    num_bond = edge_index.shape[1]
    valid_mask = edge_length <= threebody_cutoff
    ij_reverse_map = np.where(valid_mask)[0]
    edge_index = edge_index[:, valid_mask]
    if edge_index.shape[1] > 0:
        sent_indices = edge_index[0].reshape(-1, 1)
        all_indices = np.arange(num_atoms)
        n_bond_per_atom = np.count_nonzero(sent_indices == all_indices, axis=0)
        n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
        n_triple = np.sum(n_triple_i)
        n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)
        triple_bond_indices = np.zeros((n_triple, 2), dtype=np.int32)
        start = 0
        cs = 0
        for n in n_bond_per_atom:
            if n > 0:
                """
                triple_bond_indices is generated from all pair permutations of atom indices. The
                numpy version below does this with much greater efficiency. The equivalent slow
                code is:

                ```
                for j, k in itertools.permutations(range(n), 2):
                    triple_bond_indices[index] = [start + j, start + k]
                ```
                """
                r = np.arange(n)
                x, y = np.meshgrid(r, r, indexing="xy")
                c = np.stack([y.ravel(), x.ravel()], axis=1)
                final = c[c[:, 0] != c[:, 1]]
                triple_bond_indices[start : start + (n * (n - 1)), :] = final + cs
                start += n * (n - 1)
                cs += n
        triple_bond_indices = ij_reverse_map[triple_bond_indices].astype(np.int32)
        n_triple_s = np.array([np.sum(n_triple_i[0:num_atoms])])
        n_triple_ij_new = np.zeros(num_bond, dtype=np.int32)
        n_triple_ij_new[ij_reverse_map] = n_triple_ij
        n_triple_ij = n_triple_ij_new
    else:
        triple_bond_indices = np.reshape(np.array([], dtype="int32"), [-1, 2])
        if num_bond == 0:
            n_triple_ij = np.array([], dtype="int32")
        else:
            n_triple_ij = np.array([0] * num_bond, dtype="int32")
        n_triple_i = np.array([0] * num_atoms, dtype="int32")
        n_triple_s = np.array([0], dtype="int32")
    return triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s

class ASEConvertor(object):
    """
    Convert an ase.Atoms into a Graph (torch_geometric.data.Data)
    """
    def __init__(
        self,
        twobody_cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
    ):
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff
    
    def convert(
        self,
        atoms: Atoms,
        energy: float,
        force: np.ndarray,
        stress: np.ndarray,
        pbc: bool = True,
    ):
        ## Add Normalization to Atomic Positions
        scaled_positions = atoms.get_scaled_positions()
        scaled_positions = np.mod(scaled_positions, 1)
        atoms.set_scaled_positions(scaled_positions)
        
        graph_info = {}
        graph_info["num_atoms"] = len(atoms)
        graph_info["num_nodes"] = len(atoms)
        graph_info['atom_attr']=torch.FloatTensor(atoms.get_atomic_numbers()).unsqueeze(-1)
        graph_info['atom_pos']=torch.FloatTensor(atoms.get_positions())
        graph_info["cell"]=torch.FloatTensor(np.array(atoms.cell)).unsqueeze(0)
        
        sent_index, receive_index, shift_vectors, distances = get_fixed_radius_bonding(atoms, self.twobody_cutoff, pbc=pbc)
        graph_info['num_bonds']=len(sent_index)
        graph_info['edge_index']=torch.from_numpy(np.array([sent_index,receive_index]))
        graph_info["pbc_offsets"]=torch.FloatTensor(shift_vectors)
        
        triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s = compute_threebody_indices(
            num_atoms = len(atoms),
            edge_index = np.array([sent_index,receive_index]),
            edge_length = distances,
            threebody_cutoff = self.threebody_cutoff,
        )
        graph_info['three_body_indices']=torch.from_numpy(triple_bond_indices).to(torch.long)  ## [num_three_body,2]
        graph_info['num_three_body']=graph_info['three_body_indices'].shape[0]
        graph_info['num_triple_ij']=torch.from_numpy(n_triple_ij).to(torch.long).unsqueeze(-1)
        if energy is not None:
            graph_info["energy"] = torch.FloatTensor([energy])
        if force is not None:
            graph_info["force"] = torch.FloatTensor(force)
        if stress is not None:
            graph_info["stress"] = torch.FloatTensor(stress).unsqueeze(0)
        return Data(**graph_info)
    

def build_dataloader(
    atoms_list: list[Atoms],
    energy_list: list[float] = None,
    force_list: list[np.ndarray] = None,
    stress_list: list[np.ndarray] = None,
    pbc: bool = True, 
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    batch_size: int = 64,
    shuffle: bool = True,
    pin_memory: bool = False,
    only_inference: bool = False,
    num_workers: int = 0,
    verbose: bool = True,
):
    if only_inference is False:
        assert (energy_list is not None), "Energy Labels must be Provided if not for Only-Inference"
        assert (force_list is not None), "Force Labels must be Provided if not for Only-Inference"
        assert (stress_list is not None), "Stress Labels must be Provided if not for Only-Inference"
    if stress_list is not None:
        assert np.array(stress_list[0]).shape == (3, 3), "Stress must be a 3x3 Matrix"
        
    if not only_inference:
        assert len(atoms_list) == len(energy_list), "Length of Atoms and Energy Labels must be the same"
        assert len(atoms_list) == len(force_list), "Length of Atoms and Force Labels must be the same"
        assert len(atoms_list) == len(stress_list), "Length of Atoms and Stress Labels must be the same"
    
    length = len(atoms_list)
    if energy_list is None:
        energy_list = [None] * length
    if force_list is None:
        force_list = [None] * length
    if stress_list is None:
        stress_list = [None] * length
    
    graph_list = []
    convertor = ASEConvertor(
        twobody_cutoff = cutoff,
        threebody_cutoff = threebody_cutoff,
    )
    if verbose:
        for atoms, energy, force, stress in tqdm(zip(atoms_list, energy_list, force_list, stress_list), total=length, desc="Building Graphs"):
            graph = convertor.convert(atoms, energy, force, stress, pbc)
            graph_list.append(graph)
    else:
        for atoms, energy, force, stress in zip(atoms_list, energy_list, force_list, stress_list):
            graph = convertor.convert(atoms, energy, force, stress, pbc)
            graph_list.append(graph)
        
    dataloader = DataLoader(
        graph_list,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader