import torch
import torch.nn as nn
from typing import Sequence, Callable
from itertools import chain
from torch import Tensor
import torchdiffeq as tdeq


class NN(nn.Module):
    def __init__(self, device):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.Softplus(beta=20, threshold=20),
            nn.Linear(128, 128),
            nn.Softplus(beta=20, threshold=20),
            nn.Linear(128, 128),
            nn.Softplus(beta=20, threshold=20),
            nn.Linear(128, 2),
        ).to(device=device)

    def forward(self, x, t):
        return self.layers(x)


class ODEFuncBlock(nn.Module):
    def __init__(
        self,
        h_k,
        model: nn.Module,
        sigma: float,
        use_fd=False,
        ode_solver="rk4",
        device="cpu",
        n_eps=1,
    ):
        """ODE block using model, time independent

        Args:
            h_k (int): number of split steps
            model (nn.Module): model for f
            sigma (float): finite difference sigma
            use_fd (bool, optional): use finite difference. Defaults to False.
        """
        super().__init__()
        self.model = model
        self.h_steps = 3
        self.h_k = h_k
        self.sigma = sigma
        self.use_fd = use_fd
        self.ode_solver = ode_solver
        self.device = device
        self.n_eps = n_eps

    def _fd_div_at(self, x, t, f_m=None):
        """Finite difference hutchinson divergence at(x,t). Returns (B,1)
        f_m is if already evaluated f(x, t)
        """
        with torch.no_grad():
            rms = x.pow(2).mean().sqrt().clamp_min(1e-3)
        sigma_base = self.sigma if self.sigma > 0 else 1e-2
        sigma = torch.maximum(
            torch.tensor(1e-4, device=x.device, dtype=x.dtype), rms * sigma_base
        )

        acc = 0.0
        self._eps_cache = [torch.randn_like(x) for _ in range(self.n_eps)]
        for eps in self._eps_cache:
            f_p = self.model(x + sigma * eps, t)  # (B, D)
            if f_m is None:
                f_m = self.model(x, t)

            jvp = (f_p - f_m) / (sigma)  # J_f @ e
            acc += (jvp * eps).sum(dim=-1)
        return acc / len(self._eps_cache)

    def _rhs(self, t, state):
        """torchdiffeq expects (t,state) where state = (x, logdet, jacint)"""
        x, logdet, _ = state
        vfield = self.model(x, t)
        div = self._fd_div_at(x, t, f_m=vfield)

        dlogdet_dt = -div
        djac_dt = torch.zeros_like(logdet)
        return (vfield, dlogdet_dt, djac_dt)

    def forward(self, x0, t=0.0, reverse=False, full_traj=False):
        t_start, t_end = float(t), float(t) + float(self.h_k)
        t_grid = torch.linspace(
            t_start, t_end, self.h_steps + 1, dtype=x0.dtype, device=self.device
        )
        if reverse:
            t_grid = t_grid.flip(0)

        B, D = x0.shape
        logdet0 = torch.zeros(B, device=self.device, dtype=x0.dtype)
        jac0 = torch.zeros(B, device=self.device, dtype=x0.dtype)

        xT, logdetT, jacT = tdeq.odeint(
            self._rhs, (x0, logdet0, jac0), t_grid, method=self.ode_solver
        )

        if full_traj:
            return xT, logdetT, jacT
        return xT[-1], logdetT[-1], jacT[-1]


class JKO(nn.Module):
    def __init__(self, blocks: list[ODEFuncBlock]):
        super().__init__()
        if len(blocks) == 0:
            raise ValueError("Length of blocks cannot be zero.")

        self.blocks = nn.ModuleList(blocks)
        self.block_length = len(blocks)
        self.device = self.blocks[0].device

    def forward(self, x, t, block_idx=None, reverse=False):
        if block_idx is None:
            block_order = self.blocks if not reverse else reversed(self.blocks)
            for block in block_order:
                x, div_f, _ = block(x, t, reverse)
            return x, None

        out, div_f, _ = self.blocks[block_idx](x, t, reverse)
        return out, div_f

    def __len__(self):
        return self.block_length

    def get_total_time(self):
        return sum([block.h_k for block in self.blocks])


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

    def forward(
        self, x, t=0.0, reverse=False, all_block_outputs=False, full_traj=False
    ):
        """OTFlow model forward propagation

        Args:
            x (Tensor): Data
            t (float, optional): time value. Defaults to 0.0.
            reverse (bool, optional): Whether to go in reverse order. Defaults to False.
            all_block_outputs (bool, optional): Whether to return output of all blocks. Defaults to False.
            full_traj (bool, optional): Whether to return full trajectory per block. Defaults to False.

        Returns:
            Tensor: output
        """
        blocks = self.get_blocks(reverse)
        all_blocks_ret = [x]
        for idx, block in enumerate(blocks):
            is_reverse = idx >= len(self.jko1)
            x, _, _ = block(x, reverse=is_reverse, full_traj=full_traj)
            if all_block_outputs:
                all_blocks_ret.append(x)

        if all_block_outputs:
            return torch.tensor(
                torch.stack(all_blocks_ret)
            )  # of size (len(blocks) + 1, D)

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
