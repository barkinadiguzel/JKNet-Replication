import torch.nn as nn
from ..encoder.graph_encoder import GraphEncoder
from ..jumping_knowledge.jk_selector import get_jk

class JKNet(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, jk_mode):
        super().__init__()

        self.encoder = GraphEncoder(num_layers, in_dim, hidden_dim)
        self.jk = get_jk(jk_mode, hidden_dim)

    def forward(self, x, adj_norm):
        layer_outputs = self.encoder(x, adj_norm)
        return self.jk(layer_outputs)
