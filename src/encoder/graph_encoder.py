import torch.nn as nn
from ..layers.gnn_layer import GNNLayer

class GraphEncoder(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GNNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))

    def forward(self, x, adj_norm):
        outputs = []

        for layer in self.layers:
            x = layer(x, adj_norm)
            outputs.append(x)

        return outputs
