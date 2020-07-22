# Standard Library
from functools import reduce
from operator import mul
from math import floor

# Third Party
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size


class LSTM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        self.lstm = nn.LSTM(in_channels, out_channels, batch_first=True, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h_n, c_n) = self.lstm(x)
        return h_n


# TODO: Let edge_nn be lambda x: x by default, and handle edge_attr = None
# TODO: Allow LSTMConv layers to be stacked
class LSTMConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_nn: nn.Module,
                 aggr: str = "add", root_lstm: bool = True, bias: bool = True, **kwargs):
    """
    TODO: Write docstring
    
    Parameters
    ----------
    in_channels : int
        ...
    out_channels : int
        ...
    edge_nn : nn.Module
        ...
    aggr : string
        ...
    root_lstm : boolean
        ...
    bias : boolean
        ...
    """
    super(LSTMConv, self).__init__(aggr=aggr)
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.edge_nn, self.aggr = edge_nn, aggr

    self.message_lstm = LSTM(in_channels, out_channels, **kwargs)
    if root_weight:
        self.root = LSTM(in_channels, out_channels, **kwargs)
    else:
        self.register_parameter("root", None)
    
    if bias:
        self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
        self.register_parameter("bias", None)
    
    self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_nn)
        reset(self.message_lstm)
        if self.root:
            reset(self.root)
        zeros(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: OptTensor = None, 
                size: Size = None) -> torch.Tensor:
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        size = (x.size(0), x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
 
    def message(self, x_j: torch.Tensor, edge_attr: OptTensor):
        edge_weight = self.edge_nn(edge_attr)
        edge_weight = weight.view(-1, self.in_channels, self.out_channels)
        
        x_j_prime = self.message_lstm(x_j.view(1, *x_j.shape))
        return torch.matmul(x_j_prime, weight).squeeze(1)
 
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        if self.bias:
            aggr_out += self.bias
        
        return self.root_conv(x.view(1, *x_j.shape)) + aggr_out
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


# TEMP: For testing purposes only
if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.tensor([
        [[-1], [12], [7], [3]]
        [[4], [6], [1], [2]],
        [[0], [1], [1], [5]]
    ], dtype=torch.float)
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    edge_attr = torch.tensor([[0.35], [0.75], [0.12], [0.98]])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    conv = LSTMConv(in_channels=1, out_channels=10, edge_nn=nn.Linear(1, 10))
    output = conv(data)
    print(output)