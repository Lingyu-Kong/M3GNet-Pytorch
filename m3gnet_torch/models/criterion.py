import torch
import torch.nn as nn

class MSELossAndMAE(nn.Module):
    def __init__(
        self,
        energy_loss_weight: float = 1.0,
        force_loss_weight: float = 1.0,
        stress_loss_weight: float = 0.1,
    ):
        super(MSELossAndMAE, self).__init__()
        assert energy_loss_weight > 0, "Energy Loss Weight should be greater than 0"
        self.energy_loss_weight = energy_loss_weight
        self.force_loss_weight = force_loss_weight
        self.stress_loss_weight = stress_loss_weight
        
    def forward(
        self,
        energies_peratom_pred: torch.Tensor,
        forces_pred: torch.Tensor,
        stresses_pred: torch.Tensor,
        energies_peratom_target: torch.Tensor,
        forces_target: torch.Tensor,
        stresses_target: torch.Tensor,
    ):
        loss = nn.MSELoss()(energies_peratom_pred, energies_peratom_target) * self.energy_loss_weight
        mae_e = nn.L1Loss()(energies_peratom_pred, energies_peratom_target)
        if self.force_loss_weight > 0:
            loss += nn.MSELoss()(forces_pred, forces_target) * self.force_loss_weight
            mae_f = nn.L1Loss()(forces_pred, forces_target)
        else:
            mae_f = torch.tensor(0)
        if self.stress_loss_weight > 0:
            loss += nn.MSELoss()(stresses_pred, stresses_target) * self.stress_loss_weight
            mae_s = nn.L1Loss()(stresses_pred, stresses_target)
        else:
            mae_s = torch.tensor(0)
        return loss, mae_e, mae_f, mae_s
        