import torch

def normalize_adjacency(A):
    I = torch.eye(A.size(0), device=A.device)
    A_hat = A + I

    D = torch.diag(torch.pow(A_hat.sum(1), -0.5))
    return D @ A_hat @ D
