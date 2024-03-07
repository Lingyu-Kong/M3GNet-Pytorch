import torch
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
import ase.neighborlist as nl
from m3gnet_torch.data_loader.ase_loader import build_dataloader
from m3gnet_torch.potential import Potential
from ase.constraints import full_3x3_to_voigt_6_stress
from typing import Optional

def msd_compute(atoms1, atoms2):  
    if len(atoms1) != len(atoms2):  
        raise ValueError("Both Atoms objects should have the same number of atoms")  
    displacements = atoms2.positions - atoms1.positions  
    squared_displacements = np.square(displacements)  
    msd = np.sum(squared_displacements) / len(atoms1)  
    return msd  

def rdf_compute(atoms: Atoms, r_max, n_bins, elements=None):  
    scaled_pos = atoms.get_scaled_positions()  
    atoms.set_scaled_positions(np.mod(scaled_pos, 1))  
  
    num_atoms = len(atoms)  
    volume = atoms.get_volume()  
    density = num_atoms / volume  
  
    send_indices, receive_indices, distances = nl.neighbor_list(  
        'ijd',  
        atoms,  
        r_max,  
        self_interaction=False,   
    )  
  
    if elements is not None and len(elements) == 2:  
        species = np.array(atoms.get_chemical_symbols())  
        indices = np.where(np.logical_and(species[send_indices] == elements[0], species[receive_indices] == elements[1]))[0]   
        distances = distances[indices]  
  
        num_atoms = (species == elements[0]).sum()  
        density = num_atoms / volume  
  
    hist, bin_edges = np.histogram(distances, range=(0, r_max), bins=n_bins)  
    rdf_x = 0.5 * (bin_edges[1:] + bin_edges[:-1])  
    bin_volume = (4 / 3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)  
    rdf_y = hist / (bin_volume * density * num_atoms)  
  
    rdf = np.vstack((rdf_y, rdf_x)).reshape(1, 2, -1)  
    return rdf 

class AseCalculator(Calculator):
    """
    Wrap Potential class to be an ASE calculator
    """
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    def __init__(
        self,
        potential: Potential,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        stress_weight: float = 1/160.21766208, ## Convert GPa to eV/A^3
        pbc: bool = True,
    ):
        super().__init__()
        self.potential = potential.to(device)
        self.device = device
        self.stress_weight = stress_weight
        self.pbc = pbc
        
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        
        cutoff = self.potential.kernel_model.cutoff
        threebody_cutoff = self.potential.kernel_model.threebody_cutoff

        dataloader = build_dataloader(
            atoms_list=[atoms],
            energy_list=None,
            force_list=None,
            stress_list=None,
            pbc=self.pbc,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            batch_size=1,
            shuffle=False,
            only_inference=True,
            verbose=False,
        )
        for graph_batch in dataloader:
            graph_batch = graph_batch.to(self.device)
            output = self.potential(graph_batch, include_forces=True, include_stresses=True)
            self.results.update(
                energy=output['energies'].detach().cpu().numpy()[0],
                free_energy=output['energies'].detach().cpu().numpy()[0],
                forces=output['forces'].detach().cpu().numpy(),
                stress=self.stress_weight * full_3x3_to_voigt_6_stress(output['stresses'].detach().cpu().numpy()[0])
            )
        # ## Clear cuda cache
        # torch.cuda.empty_cache()