import torch

class JKMaxPool:
    def __call__(self, layer_outputs):
        return torch.stack(layer_outputs, dim=0).max(dim=0)[0]
