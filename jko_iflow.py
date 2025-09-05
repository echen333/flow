import torch
import torch.nn as nn
from torch import Tensor
from PIL import Image
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset, IterableDataset
import os
import torchdiffeq as tdeq
import matplotlib.pyplot as plt
import time
from operator import itemgetter


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

    def forward(self, x0, t=0.0, reverse=False):
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


def get_JKO(h_0, h_max, rho, num_blocks, sigma_0, d, device="cpu"):
    h_ks = [min(h_max, h_0 * (rho**idx)) for idx in range(num_blocks)]
    method = "rk4"
    flow = JKO(
        [
            ODEFuncBlock(
                h_ks[idx],
                NN(device),
                sigma_0 / np.sqrt(d),
                device=device,
                ode_solver=method,
            )
            for idx in range(num_blocks)
        ]
    )
    return flow


def sample_points_from_image(image_path, num_points=10000):
    """
    Data generation of distribution from image, fit to [-4,4].
    Taken from https://github.com/hamrel-cxu/JKO-iFlow
    """
    image = Image.open(image_path)
    # not sure where transforms but grayscale image
    image_mask = np.array(image.rotate(180).transpose(0).convert("L"))

    w, h = image.size
    x = np.linspace(-4, 4, w)
    y = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))
    means = np.concatenate([xx, yy], 1)

    img = image_mask.max() - image_mask
    probs = img.reshape(-1) / img.sum()
    std = np.array([8 / w / 2, 8 / h / 2])
    inds = np.random.choice(int(probs.shape[0]), num_points, p=probs)
    m = means[inds]
    samples = np.random.randn(*m.shape) * std + m
    return torch.tensor(samples, dtype=torch.float32)


def train_flow(
    flow: JKO,
    train_dataset: Tensor,
    num_epochs: int,
    *,
    lr: float,
    batch_size: int,
    clip_grad: bool,
    save_path: str,
    device="cpu",
):
    """Train CNF block by block"""

    prev_points: Tensor = train_dataset # (N,D)

    for block_idx in range(flow.block_length):
        start_time = time.time()
        block = flow.blocks[block_idx]
        optimizer = torch.optim.Adam(block.parameters(), lr=lr)

        epoch_data_loader = DataLoader(prev_points, batch_size)
        for epoch_idx in range(num_epochs):
            for batch_idx, batch in enumerate(epoch_data_loader):
                optimizer.zero_grad()
                out, div_f = flow(batch, t=0, block_idx=block_idx)

                V_loss = 0.5 * out.pow(2).sum(dim=-1).mean()
                delta = out - batch
                W_loss = 0.5 / (block.h_k) * delta.pow(2).sum(dim=-1).mean()
                div_loss = div_f.mean()
                loss = V_loss + div_loss + W_loss

                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                optimizer.step()
                if (epoch_idx + 1) % 50 == 0:
                    print(
                        f"block {block_idx} total: {loss.item():.2f} V_loss: {V_loss.item():.2f} W_loss: {W_loss.item():.2f} div: {div_loss.item():.2f}"
                    )

        pushed_batches = []
        flow.eval()
        for batch in epoch_data_loader:
            with torch.no_grad():
                new_points, _ = flow(batch, t=0, block_idx=block_idx)
                pushed_batches.append(new_points.detach())

        prev_points = torch.concat(pushed_batches)
        flow.train()
        print(f"Block training finished in {time.time() - start_time:.2f} seconds")

        # save block parameters
        args = {
            "lr": lr,
            "batch_size": batch_size,
            "save_path": save_path,
            "device": device
        }
        sdict = {
            "model": flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args,
        }
        os.makedirs("chkpt", exist_ok=True)
        torch.save(sdict, f"chkpt/{save_path}_{block_idx}.pth")
        plt.scatter(
            prev_points[:, 0].cpu().detach().numpy(),
            prev_points[:, 1].cpu().detach().numpy(),
        )
        plt.savefig(f"assets/block_{block_idx}.png")
        plt.close()


def reparameterize_trajectory(
    model: JKO,
    points: torch.Tensor,
    num_iters=1,
    ema_parameter: float = 0.3,
    h_max: float = 5,
):
    """Trajectory reparameterization as written in Section B.3.1"""
    assert 0 < ema_parameter and ema_parameter < 1, "ema parameter eta must be in (0,1)"
    for iter in range(num_iters):
        arc_lengths = []
        prev_points: torch.Tensor = points  # (B, D)
        for idx, block in enumerate(model.blocks):
            cur_points, _ = model(prev_points, t=0.0, block_idx=idx)  # (B, D)
            norms = torch.norm(cur_points - prev_points, dim=-1)  # (B,)
            arc_lengths.append(norms.mean())
            prev_points = cur_points
        arc_lengths = torch.tensor(arc_lengths).cpu().numpy()
        mean_arc = np.mean(arc_lengths)
        old_hk = np.array([block.h_k for block in model.blocks])
        new_hk = np.minimum(
            old_hk + ema_parameter * (mean_arc * old_hk / arc_lengths - old_hk),
            h_max * np.ones_like(old_hk),
        )
        for block, h_k in zip(model.blocks, new_hk):
            block.h_k = h_k
        print(f"mean arc length {mean_arc} std {arc_lengths.std()}")
    # TODO: do we need to train a free block at end?


class ResamplingDataset(IterableDataset):
    def __init__(self, sampling_fn, num_epochs, device="cpu"):
        super().__init__()
        self.sampling_fn = sampling_fn
        self.num_epochs = num_epochs
        self.device = device

    def __len__(self):
        return self.num_epochs

    def __iter__(self):
        for _ in range(self.num_epochs):
            yield self.sampling_fn().to(self.device)


def main():
    path = "config_rose.yaml"
    with open(path, "r") as f:
        args = yaml.safe_load(f)
    print(args)

    torch.random.manual_seed(args["seed"])
    points = sample_points_from_image(args["image_path"], args["train"]["num_points"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h_ks = [
        min(args["h_max"], args["h_0"] * (args["rho"] ** idx))
        for idx in range(args["num_blocks"])
    ]
    method = "rk4"
    flow = JKO(
        [
            ODEFuncBlock(
                h_ks[idx],
                NN(device),
                args["sigma_0"] / np.sqrt(args["d"]),
                device=device,
                ode_solver=method,
            )
            for idx in range(args["num_blocks"])
        ]
    )

    points = points.to(device=device)
    train_args = args["train"]
    train_dataset = ResamplingDataset(
        lambda: sample_points_from_image(
            args["image_path"], args["train"]["num_points"]
        ),
        train_args["epochs_per_block"],
        "cpu",
    )
    lr, batch_size, clip_grad, save_path = itemgetter(
        "lr", "batch_size", "clip_grad", "save_path"
    )(args["train"])
    print("training flow...")
    train_flow(
        flow,
        train_dataset,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )
    print(f"h_k before {[block.h_k for block in flow.blocks]}")
    reparameterize_trajectory(flow, points, 10, args["ema_parameter"], args["h_max"])
    h_ks = [block.h_k for block in flow.blocks]
    print(f"h_k after {h_ks}")
    train_flow(
        flow,
        train_dataset,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )

    sdict = {
        "model": flow.state_dict(),
        "args": args,
        "h_ks": h_ks,
    }
    os.makedirs("chkpt", exist_ok=True)
    torch.save(sdict, f"chkpt/{train_args['save_path']}.pth")


if __name__ == "__main__":
    main()
