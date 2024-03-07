from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import torch
from m3gnet_torch.potential import Potential
from m3gnet_torch.extension.ase_ext import AseCalculator, msd_compute, rdf_compute
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
import os


def main(args):
    atoms = read(args.init_struct)
    atoms.pbc = True
    potential = Potential.load(args.potential_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    init_atoms_copy = copy.deepcopy(atoms)
    calc = AseCalculator(potential, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), pbc = True)
    atoms.set_calculator(calc)
    MaxwellBoltzmannDistribution(atoms, args.temperature * units.kB)
    dyn = Langevin(
        atoms = atoms, 
        timestep = args.timestep * units.fs, 
        temperature_K = args.temperature, 
        friction = 0.002,
        trajectory = None)
    md_traj = []
    msd_traj = []
    pbar = tqdm(total=args.n_steps, desc="Langevin Dynamics, Step: ..., Temperature: ... K")
    for i in range(args.n_steps):
        try:
            dyn.run(1)
            md_traj.append(atoms.copy())
            msd = msd_compute(init_atoms_copy, atoms)
            msd_traj.append(msd)
            pbar.update(1)
            pbar.set_description(f"Langevin Dynamics, Step: {i}, Temperature: {atoms.get_temperature():.2f} K")
        except:
            break
    
    os.makedirs("md_results", exist_ok=True)
    
    write("./md_results/md_traj.xyz", md_traj)
    plt.figure()
    plt.plot(np.arange(len(md_traj)), [atoms.get_temperature() for atoms in md_traj])
    plt.xlabel("Step")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Evolution")
    plt.savefig("./md_results/md_temperature.png")
    
    plt.figure()
    plt.plot(np.arange(len(msd_traj)), msd_traj)
    plt.xlabel("Step")
    plt.ylabel("MSD (A^2)")
    plt.title("Mean Square Displacement")
    plt.savefig("./md_results/msd.png")
    
    rdf_begin_list = []
    rdf_end_list = []
    for i in range(0, int(len(md_traj)*0.1)):
        rdf = rdf_compute(md_traj[i], r_max=10, n_bins=100)
        rdf_begin_list.append(rdf)
    for i in range(int(len(md_traj)*0.9), len(md_traj)):
        rdf = rdf_compute(md_traj[i], r_max=10, n_bins=100)
        rdf_end_list.append(rdf)
    rdf_begin = np.mean(rdf_begin_list, axis=0)
    rdf_end = np.mean(rdf_end_list, axis=0)
    plt.figure()
    plt.plot(rdf_begin[0, 1], rdf_begin[0, 0], label="Begin")
    plt.plot(rdf_end[0, 1], rdf_end[0, 0], label="End")
    plt.xlabel("r (A)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.legend()
    plt.savefig("./md_results/rdf.png")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3GNet MD Example")
    parser.add_argument("--init_struct", type=str, default="./data/POSCAR", help="Initial Structure")
    parser.add_argument("--potential_path", type=str, default="./checkpoints/m3gnet_Li.pth", help="Path to the potential model")
    parser.add_argument("--temperature", type=float, default=800, help="Temperature in K")
    parser.add_argument("--timestep", type=float, default=1, help="Timestep in fs")
    parser.add_argument("--n_steps", type=int, default=100000, help="Number of steps")
    args = parser.parse_args()
    
    main(args)