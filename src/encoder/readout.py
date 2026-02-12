import torch

def global_sum_pool(x):
    return torch.sum(x, dim=0)
