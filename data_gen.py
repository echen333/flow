import torch
from torch import Tensor


def get_gaussian(mu: Tensor, sigma, num_samples, device="cpu"):
    x = torch.randn(num_samples, *mu.shape) * sigma + mu
    return x.to(device)
