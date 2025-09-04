import torch
import torch.nn as nn
from jko_iflow import JKO
from typing import Sequence, Callable


class MLP(nn.Module):
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
            layers.append(nn.Linear(in_dim, out_dim))
            if idx != len(dimensions) - 2:
                layers.append(self.activation_fn)
        self.layers = nn.Sequential(*layers)

    # classmethod?


class OTFlow(nn.Module):
    def __init__(self, jko1: JKO, jko2: JKO):
        super().__init__()
        self.jko1 = jko1
        self.jko2 = jko2

    def forward(self, x, t=0.0, reverse=False):
        if not reverse:
            x1 = self.jko1(x, t)
            t1 = self.jko1.get_total_time()
            x2 = self.jko2(x1, t1, reverse=True)
        else:
            x1 = self.jko2(x, t)
            t1 = self.jko2.get_total_time()
            x2 = self.jko1(x1, t1, reverse=True)
        return x2
