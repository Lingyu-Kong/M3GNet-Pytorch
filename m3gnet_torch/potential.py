import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_ema import ExponentialMovingAverage
from m3gnet_torch.models.criterion import MSELossAndMAE
from m3gnet_torch.models.m3gnet import M3GNet
from tqdm import tqdm
from typing import Optional, List
import os

class Potential(nn.Module):
    """
    Potential Wrapper for M3GNet Model
    """
    def __init__(
        self,
        kernel_model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(Potential, self).__init__()
        self.kernel_model = kernel_model
        self.ema = ExponentialMovingAverage(self.kernel_model.parameters(), decay=0.99)
        self.device = device
        self.to(device)
        
    def forward(
        self,
        input_batch,
        include_forces: bool = True, 
        include_stresses: bool = True
    ):
        output = {}
        strain = torch.zeros_like(input_batch.cell, device=self.device) ## Strain is the Operation on the Cell
        volume = torch.linalg.det(input_batch.cell)
        if include_forces:
            input_batch.atom_pos.requires_grad_(True)
        if include_stresses is True:
            strain.requires_grad_(True)
            input_batch.cell = torch.matmul(input_batch.cell, (torch.eye(3, device=self.device)[None, ...] + strain))
            strain_augment = torch.repeat_interleave(strain, input_batch.num_atoms, dim=0)
            input_batch.atom_pos = torch.einsum("bi, bij -> bj", input_batch.atom_pos, (torch.eye(3, device=self.device)[None, ...] + strain_augment))
            
        energies = self.kernel_model.forward(input_batch)
        output['energies'] = energies
        
        if include_forces is True and include_stresses is False:

            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energies, )]
            grad = torch.autograd.grad(outputs = [energies, ], inputs = [input_batch.atom_pos], grad_outputs=grad_outputs, create_graph=self.kernel_model.training)

            # Dump out gradient for forces
            force_grad = grad[0]
            if force_grad is not None:
                forces = torch.neg(force_grad)
                output['forces'] = forces

        # Take derivatives up to second order if both forces and stresses are required
        if include_forces is True and include_stresses is True:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energies, )]
            grad = torch.autograd.grad(outputs = [energies, ], inputs = [input_batch.atom_pos, strain], grad_outputs=grad_outputs, create_graph=self.kernel_model.training)

            # Dump out gradient for forces and stresses
            force_grad = grad[0]
            stress_grad = grad[1]

            if force_grad is not None:

                forces = torch.neg(force_grad)
                output['forces'] = forces

            if stress_grad is not None:

                stresses = 1 / volume[:, None, None] * stress_grad * 160.21766208 ## Convert Ev/Ang^3 to GPa
                output['stresses'] = stresses
        
        if include_forces is False:
            output['forces'] = None
        if include_stresses is False:
            output['stresses'] = None

        return output

        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        energy_loss_weight: float = 1.0,
        force_loss_weight: float = 1.0,
        stress_loss_weight: float = 0.1,
        epochs: int = 100,
        lr: float = 1e-3,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_schedule: bool = True,
        gamma: float = 0.5,
        early_stop_patience: int = 100,
        ckpt_save_path: str = "./checkpoints",
    ):
        optimizer = optimizer(self.kernel_model.parameters(), lr=lr)
        if val_loader is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=gamma, verbose=True)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        self.kernel_model.train()
        criterion = MSELossAndMAE(energy_loss_weight=energy_loss_weight, force_loss_weight=force_loss_weight, stress_loss_weight=stress_loss_weight)
        include_forces = force_loss_weight > 0
        include_stresses = stress_loss_weight > 0
        best_val_loss = float("inf")
        early_stop_wait = 0
        self.kernel_model.train()
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            loss_list = []
            mae_e_list = []
            mae_f_list = []
            mae_s_list = []
            pbar = tqdm(train_loader, desc="Train MAE_E: ......, Train MAE_F: ......, Train MAE_S: ......")
            for batch_num, batch_data in enumerate(pbar):
                optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                output = self.forward(batch_data, include_forces=include_forces, include_stresses=include_stresses)
                energies_pred = output['energies']
                energies_peratom_pred = energies_pred / batch_data.num_atoms
                forces_pred = output['forces']
                stresses_pred = output['stresses']
                energies_peratom_target = batch_data.energy / batch_data.num_atoms
                forces_target = batch_data.force
                stresses_target = batch_data.stress
                loss, mae_e, mae_f, mae_s = criterion(energies_peratom_pred, forces_pred, stresses_pred, energies_peratom_target, forces_target, stresses_target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.kernel_model.parameters(), 1., norm_type=2)
                optimizer.step()
                self.ema.update()
                
                loss_list += [loss.item()]*batch_data.num_graphs
                mae_e_list += [mae_e.item()]*batch_data.num_graphs
                mae_f_list += [mae_f.item()]*batch_data.num_graphs
                mae_s_list += [mae_s.item()]*batch_data.num_graphs
                
                if batch_num == len(train_loader) - 1:
                    mae_e_mean = sum(mae_e_list) / len(mae_e_list)
                    if include_forces:
                        mae_f_mean = sum(mae_f_list) / len(mae_f_list)
                    else:
                        mae_f_mean = 0
                    if include_stresses:
                        mae_s_mean = sum(mae_s_list) / len(mae_s_list)
                    else:
                        mae_s_mean = 0
                    pbar.set_description(f"Train MAE_E: {mae_e_mean:.4f}, Train MAE_F: {mae_f_mean:.4f}, Train MAE_S: {mae_s_mean:.4f}")
            
            if val_loader is not None:
                val_loss, val_mae_e, val_mae_f, val_mae_s = self.evaluate(val_loader, energy_loss_weight, force_loss_weight, stress_loss_weight)
                print(f"Validation Loss: {val_loss:.4f}, Validation MAE_E: {val_mae_e:.4f}, Validation MAE_F: {val_mae_f:.4f}, Validation MAE_S: {val_mae_s:.4f}")
            if lr_schedule and val_loader is not None:
                scheduler.step(val_loss)
            elif lr_schedule and val_loader is None:
                scheduler.step()
            print("Curent Learning Rate: ", optimizer.param_groups[0]['lr'])
                
            if val_loader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_wait = 0
                    with self.ema.average_parameters():
                        self.kernel_model.save(os.path.join(ckpt_save_path, "best_model.pth"))
                else:
                    early_stop_wait += 1
                    if early_stop_wait > early_stop_patience:
                        print(f"Early Stopping at Epoch {epoch}")
                        break
                
        self.ema.copy_to(self.kernel_model.parameters())
        torch.cuda.empty_cache()
            
    def evaluate(
        self,
        data_loader: DataLoader,
        energy_loss_weight: float = 1.0,
        force_loss_weight: float = 1.0,
        stress_loss_weight: float = 0.1,
    ):
        self.kernel_model.eval()
        include_forces = force_loss_weight > 0
        include_stresses = stress_loss_weight > 0
        criterion = MSELossAndMAE(energy_loss_weight=energy_loss_weight, force_loss_weight=force_loss_weight, stress_loss_weight=stress_loss_weight)
        with self.ema.average_parameters():
            loss_list = []
            mae_e_list = []
            mae_f_list = []
            mae_s_list = []
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                output = self.forward(batch_data, include_forces=True, include_stresses=True)
                energies_pred = output['energies']
                energies_peratom_pred = energies_pred / batch_data.num_atoms
                forces_pred = output['forces']
                stresses_pred = output['stresses']
                energies_peratom_target = batch_data.energy / batch_data.num_atoms
                forces_target = batch_data.force
                stresses_target = batch_data.stress
                loss, mae_e, mae_f, mae_s = criterion(energies_peratom_pred, forces_pred, stresses_pred, energies_peratom_target, forces_target, stresses_target)
                loss_list += [loss.item()]*batch_data.num_graphs
                mae_e_list += [mae_e.item()]*batch_data.num_graphs
                mae_f_list += [mae_f.item()]*batch_data.num_graphs
                mae_s_list += [mae_s.item()]*batch_data.num_graphs
            loss_mean = sum(loss_list) / len(loss_list)
            mae_e_mean = sum(mae_e_list) / len(mae_e_list)
            if include_forces:
                mae_f_mean = sum(mae_f_list) / len(mae_f_list)
            else:
                mae_f_mean = 0
            if include_stresses:
                mae_s_mean = sum(mae_s_list) / len(mae_s_list)
            else:
                mae_s_mean = 0
        self.kernel_model.train()
        return loss_mean, mae_e_mean, mae_f_mean, mae_s_mean
    
    def save(
        self,
        save_path: str,
    ):
        assert save_path.split(".")[-1] == "pth", "Save Path must be a .pth file"
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint = {
            "model_name": self.kernel_model.name,
            "model_params": self.kernel_model.get_model_params(),
            "state_dict": self.kernel_model.state_dict(),
            # "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, save_path)
        
    @staticmethod
    def load(
        path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        checkpoint = torch.load(path)
        if checkpoint["model_name"] == "M3GNet":
            model = M3GNet(**checkpoint["model_params"])
            model.load_state_dict(checkpoint["state_dict"])
            potential = Potential(model, device)
            # potential.optimizer.load_state_dict(checkpoint["optimizer"])
            potential.ema.load_state_dict(checkpoint["ema"])
            # potential.scheduler.load_state_dict(checkpoint["scheduler"])
            potential.kernel_model.eval()
            del checkpoint
            return potential
        
                
                
                
                