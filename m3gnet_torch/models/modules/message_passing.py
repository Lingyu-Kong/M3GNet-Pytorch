import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from m3gnet_torch.models.modules import LinearLayer, SwishLayer, SigmoidLayer, GatedMLP

def fc_function(r: torch.Tensor, cutoff: float):
    result = 1 - 6 * torch.pow(r/cutoff, 5) + 15 * torch.pow(r/cutoff, 4) - 10 * torch.pow(r/cutoff, 3)
    return result

class ManyBodyToBond(nn.Module):
    """
    Many-Body to Bond Module in the Main Block
    Equation (2)(3) in paper: https://arxiv.org/pdf/2202.02450.pdf
    """
    def __init__(
            self,
            cutoff: float,
            threebody_cutoff: float,
            hidden_dim: int,
            spherical_dim: int,
    ):
        super().__init__()
        self.node_mlp = SigmoidLayer(in_dim=hidden_dim, out_dim=spherical_dim)
        self.edge_gate_mlp = GatedMLP(in_dim=spherical_dim, out_dims=[hidden_dim], activation="swish", use_bias=False)
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        
    def forward(
        self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        three_basis: torch.Tensor,
        edge_index: torch.Tensor,
        three_body_index: torch.Tensor,
        edge_length: torch.Tensor,
        num_triple_ij: torch.Tensor,
    ):
        node_attr = self.node_mlp(node_attr)
        node2edge = node_attr[edge_index[0]][three_body_index[:, 1]] * fc_function(edge_length[three_body_index[:, 0]], cutoff=self.threebody_cutoff) * fc_function(edge_length[three_body_index[:, 1]], cutoff=self.threebody_cutoff)
        three_basis = three_basis * node2edge
        threebody2edge_index = torch.arange(edge_attr.shape[0]).to(three_basis.device)
        threebody2edge_index = threebody2edge_index.repeat_interleave(num_triple_ij)
        edge_attr_tuda = scatter_sum(three_basis, threebody2edge_index, dim=0, dim_size=edge_attr.shape[0])
        edge_attr_prime = edge_attr + self.edge_gate_mlp(edge_attr_tuda)
        return edge_attr_prime
    
class MainBlock(nn.Module):
    """
    Main Block for M3GNet
    Equation (4)(5)(6) in paper: https://arxiv.org/pdf/2202.02450.pdf
    ## (6) is not implemented since we curently don't need global state
    """
    def __init__(
        self,
        cutoff: float,
        threebody_cutoff: float,
        hidden_dim: int,
        max_n: int,
        max_l: int,
    ):
        super().__init__()
        self.manybody2bond = ManyBodyToBond(cutoff, threebody_cutoff, hidden_dim, max_l*max_n)
        self.edge02node_layer = LinearLayer(max_n, hidden_dim)
        self.node_gate_mlp = GatedMLP(in_dim=2*hidden_dim+hidden_dim,out_dims=[hidden_dim,hidden_dim],activation="swish")
        self.edge02edge_layer = LinearLayer(max_n, hidden_dim)
        self.edge_gate_mlp = GatedMLP(in_dim=2*hidden_dim+hidden_dim,out_dims=[hidden_dim,hidden_dim],activation="swish")
        
    def forward(
        self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_attr_zero: torch.Tensor,
        edge_index: torch.Tensor,
        three_basis: torch.Tensor,
        three_body_index: torch.Tensor,
        edge_length: torch.Tensor,
        num_triple_ij: torch.Tensor,
    ):
        ## Many body to bond
        edge_attr=self.manybody2bond(
            node_attr,
            edge_attr,
            three_basis,
            edge_index,
            three_body_index,
            edge_length,
            num_triple_ij.view(-1),
        )
        
        # update atom feature
        feat=torch.concat([node_attr[edge_index[0]],node_attr[edge_index[1]],edge_attr],dim=1)
        node_attr_prime=self.node_gate_mlp(feat)*self.edge02node_layer(edge_attr_zero)
        node_attr = node_attr + scatter_sum(node_attr_prime,edge_index[0],dim=0,dim_size=node_attr.shape[0])
        
        # update bond feature
        feat=torch.concat([node_attr[edge_index[0]],node_attr[edge_index[1]],edge_attr],dim=1)
        edge_attr = edge_attr + self.edge_gate_mlp(feat)*self.edge02edge_layer(edge_attr_zero)

        return node_attr, edge_attr