import torch
import torch.nn as nn
from jko_iflow import JKO, ODEFuncBlock
from typing import Sequence, Callable
from itertools import chain
from torch import Tensor


class ODEFunction(nn.Module):
    """ODE Function"""

    def __init__(self, dimensions: tuple[int], activation_fn="relu"):
        super().__init__()
        activation_mapping = {
            "relu": nn.ReLU(),
        }
        assert activation_fn in activation_mapping.keys(), (
            "invalid activation function."
        )
        self.activation_fn = activation_mapping[activation_fn]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            # need to concat time dimension
            layers.append(nn.Linear(in_dim + (1 if idx == 0 else 0), out_dim))
            if idx != len(dimensions) - 2:
                layers.append(self.activation_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x, t=0.0):
        return self.layers[torch.concat([x, t])]


class OTFlow(nn.Module):
    """OTFlow, doesn't really have the same memory efficiency as JKO"""
    
    def __init__(self, jko1: JKO, jko2: JKO):
        super().__init__()
        self.jko1 = jko1
        self.jko2 = jko2
        self.device = jko1.device

    def get_blocks(self, reverse=False) -> list[ODEFuncBlock]:
        blocks: list[ODEFuncBlock] = chain(self.jko1.blocks, reversed(self.jko2.blocks))
        if reverse:
            blocks = reversed(list(blocks))
        return blocks

    def forward(self, x, t=0.0, reverse=False):
        blocks = self.get_blocks(reverse)
        for idx, block in enumerate(blocks):
            is_reverse = (idx >= len(self.jko1))
            x, _, _ = block(x, reverse=is_reverse)
        return x

    def get_W2_loss(self, points: Tensor, reverse=False):
        prev_points = points
        blocks = self.get_blocks(reverse)

        W2_dists = torch.zeros(points.shape[0], device=self.device)
        for block in blocks:
            points, _, _ = block(points)
            delta = points - prev_points
            prev_points = points
            W2_dists += torch.norm(delta, p=2, dim=1).pow(2) / block.h_k
        return W2_dists.mean()


class BasicClassifier(nn.Module):
    def __init__(self, in_dim, device):
        super().__init__()
        # might need some regularization
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

    def forward(self, x):
        return self.model(x)
