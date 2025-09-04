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
    def __init__(self, jko1: JKO, jko2: JKO):
        super().__init__()
        self.jko1 = jko1
        self.jko2 = jko2
        self.device = jko1.device

    def forward(self, x, t=0.0, reverse=False):
        if not reverse:
            x1, _ = self.jko1(x, t)
            t1 = self.jko1.get_total_time()
            x2, _ = self.jko2(x1, t1, reverse=True)
        else:
            x1, _ = self.jko2(x, t)
            t1 = self.jko2.get_total_time()
            x2, _ = self.jko1(x1, t1, reverse=True)
        return x2

    def get_W2_loss(self, points: Tensor, reverse=False):
        prev_points = points
        blocks: list[ODEFuncBlock] = chain(self.jko1.blocks, reversed(self.jko2.blocks))
        if reverse:
            blocks = reversed(list(blocks))

        W2_dists = torch.zeros(points.shape[0], device=self.device)
        for block in blocks:
            points, _, _ = block(points)
            delta = points - prev_points
            prev_points = points
            W2_dists += torch.norm(delta, p=2, dim=1).pow(2) / block.h_k
        return W2_dists.mean()


class BasicClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # might need some regularization
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)
