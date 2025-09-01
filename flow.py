import torch
import torch.nn as nn
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
import yaml
import argparse
from torch.utils.data import DataLoader, Dataset
import os
import torchdiffeq as tdeq


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x, t):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x


class ODEFuncBlock(nn.Module):
    def __init__(
        self, h_k, model: nn.Module, sigma: float, use_fd=False, ode_solver="rk4"
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

    def vector_field(self, t, x):
        return self.model(x, t)

    def _fd_div_at(self, x, t, n_eps=1):
        """Finite difference hutchinson divergence at(x,t). Returns (B,1)"""
        divs = []
        for _ in range(n_eps):
            eps = torch.randn_like(x)  # (B, D)
            f_x = self.model(x, t)  # (B, D)
            f_x_eps = self.model(x + self.sigma * eps, t)

            div_f = ((f_x_eps - f_x) / self.sigma) * eps
            div = div_f.sum(dim=-1)  # (B, )
            divs.append(div)
        return torch.stack(divs, dim=0).mean(dim=0)

    def forward(self, x0, t=0.0, reverse=False):
        t_start, t_end = float(t), float(t) + float(self.h_k)
        t_grid = torch.linspace(t_start, t_end, self.h_steps + 1, dtype=x0.dtype)
        if reverse:
            t_grid = t_grid.flip(0)
        # breakpoint()
        x_traj = tdeq.odeint(self.vector_field, x0, t_grid, method=self.ode_solver)

        # use hutchinson trace estimator to compute div_f
        div_f = self._fd_div_at(x0, t)
        return x_traj[-1, :], div_f


class JKO(nn.Module):
    def __init__(self, blocks: list[ODEFuncBlock]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.block_length = len(blocks)

    def forward(self, x, t, block_idx=None, reverse=False):
        out, div_f = None, None
        if block_idx is None:
            block_order = self.blocks if not reverse else reversed(self.blocks)
            for block in block_order:
                x, div_f = block(x, t, reverse)
            return x, None
        else:
            out, div_f = self.blocks[block_idx](x, t, reverse)
            return out, div_f


def sample_points_from_image(image_path, num_points=100000):
    """
    Sample points from a black and white image.
    Sample points from the black part of the image.
    Returns a list of coordinates of the sampled points.
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
    means = np.concat([xx, yy], 1)

    img = image_mask.max() - image_mask
    probs = img.reshape(-1) / img.sum()
    std = np.array([8 / w / 2, 8 / h / 2])
    inds = np.random.choice(int(probs.shape[0]), num_points, p=probs)
    m = means[inds]
    samples = np.random.randn(*m.shape) * std + m
    return torch.tensor(samples, dtype=torch.float32)


def train_flow(
    flow: JKO, train_dataset: Dataset, args: dict, test_dataset: Dataset | None = None
):
    """train block by block"""
    train_args = args["train"]
    assert len(train_dataset) == train_args["num_points"]
    data_loader = DataLoader(train_dataset, train_args["batch_size"], shuffle=True)
    batches = []
    for points in data_loader:
        batches.append(points)

    import matplotlib.pyplot as plt

    for block_idx in range(flow.block_length):
        block = flow.blocks[block_idx]
        optimizer = torch.optim.Adam(block.parameters(), lr=train_args["lr"])
        for epoch_idx in range(train_args["epochs_per_block"]):
            for batch_idx, batch in enumerate(batches):
                optimizer.zero_grad()
                out, div_f = flow(batch, t=0, block_idx=block_idx)

                V_loss = 0.5 * out.pow(2).sum(dim=-1).mean()
                delta = out - batch
                # breakpoint()
                W_loss = 0.5 / (block.h_k) * delta.pow(2).sum(dim=-1).mean()
                div_loss = div_f.mean()
                loss = V_loss - div_loss + W_loss

                loss.backward()
                if train_args["clip_grad"]:
                    torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
                optimizer.step()
                if epoch_idx % 50 == 0:
                    print(
                        "loss",
                        V_loss.item(),
                        "W_loss: ",
                        W_loss.item(),
                        "div: ",
                        div_loss.item(),
                        "total: ",
                        loss,
                    )
                    print(f"Block {block_idx} loss: {loss.item()}")
                # breakpoint()

        # Use detach() to create a new tensor and avoid graph reuse
        for batch_idx, batch in enumerate(batches):
            with torch.no_grad():
                batches[batch_idx], _ = flow(batch, t=0, block_idx=block_idx)

        # save block parameters
        sdict = {
            "model": flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args,
        }
        os.makedirs("chkpt", exist_ok=True)
        torch.save(sdict, f"chkpt/{train_args['save_path']}_{block_idx}.pth")
        plt.scatter(
            batches[0][:, 0].detach().numpy(),
            batches[0][:, 1].detach().numpy(),
        )
        plt.savefig(f"block_{block_idx}.png")
        plt.close()


class MyDataset(Dataset):
    def __init__(self, points):
        super().__init__()
        self.points = points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]


def main():
    path = "config_rose.yaml"
    with open(path, "r") as f:
        args = yaml.safe_load(f)
    print(args)
    torch.random.manual_seed(args["seed"])
    points = sample_points_from_image(args["image_path"], args["train"]["num_points"])

    h_ks = [
        min(args["h_max"], args["h_0"] * (args["rho"] ** idx))
        for idx in range(args["num_blocks"])
    ]
    flow = JKO(
        [
            ODEFuncBlock(h_ks[idx], NN(), args["sigma_0"] / args["d"])
            for idx in range(args["num_blocks"])
        ]
    )

    train_dataset = MyDataset(points)
    train_flow(flow, train_dataset, args)


if __name__ == "__main__":
    main()
