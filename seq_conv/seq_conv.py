# Standard Library
from functools import reduce
from operator import mul
from math import floor

# Third Party
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset, zeros
from torch_geometric.nn.conv import MessagePassing

# TODO: Add docstrings

class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            batch_first=True,
            **kwargs
        )

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        return h_n


# TODO: Allow LSTMConv layers to be stacked
class SeqConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_nn, aggr="add",
                 root_lstm=True, bias=True, **kwargs):
        """
        TODO: Write docstring

        Parameters
        ----------
        in_channels : int
            Number of channels in the input node sequence (e.g. if each node has
            a sequence of vectors of size n associated with it, then
            in_channels = n)
        out_channels : int
            Number of channels in the output node embedding
        edge_nn : nn.Module
            A neural network h_Θ that maps edge features, edge_attr, of shape
            [-1, num_edge_features] to shape [-1, out_channels]
        aggr : string
            The message aggregation scheme to use ("add", "mean", "max")
        root_lstm : bool
            If set to False, the layer will not add the LSTM-transformed root
            node features to the output
        bias : bool
            If set to False, the layer will not learn an additive bias
        """
        super(SeqConv, self).__init__(aggr=aggr)

        self.in_channels, self.out_channels = in_channels, out_channels
        self.edge_nn, self.aggr = edge_nn, aggr

        self.message_lstm = LSTM(in_channels, out_channels, **kwargs) # phi_m
        if root_lstm:
            self.root = LSTM(in_channels, out_channels, **kwargs) # phi_r
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

    def forward(self, x, edge_index, edge_attr, size=None):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        size = (x.size(0), x.size(0))

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        edge_weight = self.edge_nn(edge_attr)
        x_j_prime = self.message_lstm(x_j).squeeze()

        return edge_weight * x_j_prime

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out += self.bias
        if self.root is not None:
            x = self.root(x)

        x_i_prime = x + aggr_out
        return x_i_prime.squeeze()

    def __repr__(self):
        return "".join([
            self.__class__.__name__,
            f"({self.in_channels}, ",
            f"{self.out_channels})"
        ])
