import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from m3gnet_torch.models.modules import SmoothBesselBasis, SphericalBasisLayer, MLP, MainBlock, GatedMLP
from m3gnet_torch.models.modules.scaler import AtomScaling

class M3GNet(nn.Module):
    """
    M3GNet Implemented with Pytorch
    Paper Reference: https://arxiv.org/pdf/2202.02450.pdf
    """
    def __init__(
        self,
        num_layers: int = 4,
        hidden_dim: int = 64,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_z: int = 94, ## Max Number of Elements
    ):
        super(M3GNet, self).__init__()
        self.name = "M3GNet"
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(in_dim=max_n, out_dims=[hidden_dim], activation="swish", use_bias=False)
        module_list = [MainBlock(cutoff, threebody_cutoff, hidden_dim, max_n, max_l) for _ in range(num_layers)]
        self.main_blocks = nn.ModuleList(module_list)
        self.final = GatedMLP(in_dim=hidden_dim, out_dims=[hidden_dim, hidden_dim, 1], activation=["swish", "swish", None])
        self.apply(self.init_weights)
        self.atom_embedding = MLP(in_dim=max_z + 1, out_dims=[hidden_dim], activation=None, use_bias=False)
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z)
        self.max_z = max_z
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.to(device)
        
    def forward(
        self,
        input_batch,
    ):
        atom_pos = input_batch.atom_pos
        cell = input_batch.cell
        pbc_offsets = input_batch.pbc_offsets
        atom_attr = input_batch.atom_attr
        edge_index = input_batch.edge_index
        three_body_indices = input_batch.three_body_indices
        num_three_body = input_batch.num_three_body
        num_triple_ij = input_batch.num_triple_ij
        num_atoms = input_batch.num_atoms
        num_bonds = input_batch.num_bonds
        num_graphs = input_batch.num_graphs
        
        ## ---------------index convertion for three_body_indices---------------##
        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        index_bias = torch.repeat_interleave(cumsum, num_three_body, dim=0).unsqueeze(-1)
        three_body_indices = three_body_indices + index_bias
        
        atoms_batch = torch.repeat_interleave(repeats=num_atoms)
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = atom_pos[edge_index[0]] - (atom_pos[edge_index[1]]
                                            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch]))
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1. + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        ## featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  ## e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        # Main Loop
        for _, main_block in enumerate(self.main_blocks):
            atom_attr, edge_attr = main_block(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_triple_ij,
            )

        energies_i = self.final(atom_attr).view(-1)  ## [batch_size*num_atoms]
        energies_i = self.normalizer(energies_i, atomic_numbers)
        energies = scatter_sum(energies_i, atoms_batch, dim=0, dim_size=num_graphs)

        return energies  ## [batch_size]
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            
    @torch.jit.export
    def one_hot_atoms(self, species):
        return F.one_hot(species, num_classes=self.max_z + 1).float()
    
    def get_model_params(self):
        return {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "max_l": self.max_l,
            "max_n": self.max_n,
            "cutoff": self.cutoff,
            "threebody_cutoff": self.threebody_cutoff,
            "max_z": self.max_z,
        }
    
    def save(
        self,
        path: str,
    ):
        model_dict = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "max_l": self.max_l,
            "max_n": self.max_n,
            "cutoff": self.cutoff,
            "threebody_cutoff": self.threebody_cutoff,
            "max_z": self.max_z,
            "state_dict": self.state_dict(),
        }
        torch.save(model_dict, path)
        
    @staticmethod
    def load(
        path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        model_dict = torch.load(path)
        model = M3GNet(
            num_layers=model_dict["num_layers"],
            hidden_dim=model_dict["hidden_dim"],
            max_l=model_dict["max_l"],
            max_n=model_dict["max_n"],
            cutoff=model_dict["cutoff"],
            threebody_cutoff=model_dict["threebody_cutoff"],
            device=device,
            max_z=model_dict["max_z"],
        )
        model.load_state_dict(model_dict["state_dict"])
        return model