# lstm-gnn

`lstm-gnn` is a PyTorch implementation of a graph convolutional operator that uses long short-term memory (LSTM) network as a filter -- that is, LSTM is used to update node embeddings. This is useful for graph datasets where each node represents a time series or sequence of vectors.

<p align="center">
    <img src="assets/equation.png" width="47%" />
</p>

Where _&phi;<sub>r</sub>_ and _&phi;<sub>m</sub>_ are LSTMs (`torch.nn.LSTM()`), and _h<sub>Θ</sub>_ is a neural network. The last hidden state of each LSTM, _h<sub>n</sub>_, is used during updating.

## Installation

`lstm-gnn` can be installed with `pip`:

```bash
$ pip install lstm-gnn
```

## Usage

This module is built on PyTorch Geometric, and inherits from the `MessagePassing` class. It takes the following parameters:

- **in_channels** (_int_): 
- **out_channels** (_int_):
- **edge_nn** (_torch.nn.Module_):
- **aggr** (_string_):
- **root_lstm** (_boolean_):
- **bias** (_boolean_):

Here's an example:

```python
import torch
from lstm_gnn import LSTMConv

# Convolutional layer
conv = LSTMConv(
    in_channels=1,
    out_channels=5,
    edge_nn=torch.nn.Linear(2, 5)
)

# Your input graph (see: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs)
x = torch.randn((3, 12, 1), dtype=torch.float) # Shape is [num_nodes, seq_len, num_features]
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)
edge_attr = torch.randn((4, 2), dtype=torch.long)

# Your output graph
x = conv(x, edge_index, edge_attr) # Shape is now [3, 5]
```