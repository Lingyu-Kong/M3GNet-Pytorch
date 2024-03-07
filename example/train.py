import os
import torch
from ase.io import read
import numpy as np
from m3gnet_torch.data_loader.ase_loader import build_dataloader
from m3gnet_torch.potential import Potential
from m3gnet_torch.models.m3gnet import M3GNet
import argparse


def main(args):
    ## ========================== Load Data ========================== ##
    train_atoms_list = read(args.train_data, index=":")
    energy_list = [atoms.get_potential_energy() for atoms in train_atoms_list]
    force_list = [np.array(atoms.get_forces()) for atoms in train_atoms_list]
    stress_list = [np.array(atoms.get_stress(voigt=False)) * 160.21766208 for atoms in train_atoms_list]
    train_loader = build_dataloader(
        atoms_list=train_atoms_list,
        energy_list=energy_list,
        force_list=force_list,
        stress_list=stress_list,
        pbc=args.pbc,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        batch_size=args.batch_size,
        shuffle=True,
    )
    if args.val_data is not None and os.path.exists(args.val_data):
        val_atoms_list = read(args.val_data, index=":")
        energy_list = [atoms.get_potential_energy() for atoms in val_atoms_list]
        force_list = [np.array(atoms.get_forces()) for atoms in val_atoms_list]
        stress_list = [np.array(atoms.get_stress(voigt=False)) * 160.21766208 for atoms in val_atoms_list]
        val_loader = build_dataloader(
            atoms_list=val_atoms_list,
            energy_list=energy_list,
            force_list=force_list,
            stress_list=stress_list,
            pbc=args.pbc,
            cutoff=args.cutoff,
            threebody_cutoff=args.threebody_cutoff,
            batch_size=8,
            shuffle=False,
        )
    else:
        val_loader = None
    test_atoms_list = read(args.test_data, index=":")
    energy_list = [atoms.get_potential_energy() for atoms in test_atoms_list]
    force_list = [np.array(atoms.get_forces()) for atoms in test_atoms_list]
    stress_list = [np.array(atoms.get_stress(voigt=False)) * 160.21766208 for atoms in test_atoms_list]
    test_loader = build_dataloader(
        atoms_list=test_atoms_list,
        energy_list=energy_list,
        force_list=force_list,
        stress_list=stress_list,
        pbc=args.pbc,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        batch_size=8,
        shuffle=False,
    )
    
    ## ========================== Build Model ========================== ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M3GNet(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        max_l=args.max_l,
        max_n=args.max_n,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        device=device,
    )
    potential = Potential(model, device)
    
    ## ========================== Train Model ========================== ##
    potential.train(
        train_loader=train_loader,
        val_loader=val_loader,
        energy_loss_weight=args.energy_loss_weight,
        force_loss_weight=args.force_loss_weight,
        stress_loss_weight=args.stress_loss_weight,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=torch.optim.Adam,
        lr_schedule=args.lr_schedule,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
    )
    
    ## ========================== Test Model ========================== ##
    test_loss, test_mae_e, test_mae_f, test_mae_s = potential.evaluate(
        data_loader=test_loader,
        energy_loss_weight=args.energy_loss_weight,
        force_loss_weight=args.force_loss_weight,
        stress_loss_weight=args.stress_loss_weight,
    )
    print("=====================================================================================================================================")
    print(f"Test Loss: {test_loss:.4f}, Test MAE Energy: {test_mae_e:.4f}, Test MAE Force: {test_mae_f:.4f}, Test MAE Stress: {test_mae_s:.4f}")
    
    potential.save("./checkpoints/m3gnet_Li.pth")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M3GNet Example")
    parser.add_argument("--train_data", type=str, default="./data/Li_sub.xyz", help="Path to the training data")
    parser.add_argument("--val_data", type=str, default=None, help="Path to the validation data")
    parser.add_argument("--test_data", type=str, default="./data/Li512_val.xyz", help="Path to the test data")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff for 2-body interactions")
    parser.add_argument("--threebody_cutoff", type=float, default=4.0, help="Cutoff for 3-body interactions")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--pbc", type=bool, default=True, help="Periodic Boundary Conditions")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in M3GNet")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden Dimension in M3GNet")
    parser.add_argument("--max_l", type=int, default=4, help="Max l in Spherical Harmonics")
    parser.add_argument("--max_n", type=int, default=4, help="Max n in Spherical Harmonics")
    parser.add_argument("--energy_loss_weight", type=float, default=1.0, help="Energy Loss Weight")
    parser.add_argument("--force_loss_weight", type=float, default=1.0, help="Force Loss Weight")
    parser.add_argument("--stress_loss_weight", type=float, default=0.1, help="Stress Loss Weight")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--lr_schedule", type=bool, default=True, help="Learning Rate Schedule")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for Learning Rate Schedule")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Early Stopping Patience")
    args = parser.parse_args()
    
    main(args)
    