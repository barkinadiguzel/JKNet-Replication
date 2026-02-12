import torch

class JKConcat:
    def __call__(self, layer_outputs):
        return torch.cat(layer_outputs, dim=-1)
