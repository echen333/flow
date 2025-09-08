import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from models import OTFlow, ODEFuncBlock
from typing import Optional
from torch import Tensor
import torch.nn as nn
import seaborn as sns


def plot_two_distributions(Q_hat, Q_data, dbg_path: Optional[str] = None, show=False):
    plt.scatter(
        Q_hat[:, 0].cpu().detach().numpy(),
        Q_hat[:, 1].cpu().detach().numpy(),
        c="green",
    )
    plt.scatter(
        Q_data[:, 0].cpu().detach().numpy(),
        Q_data[:, 1].cpu().detach().numpy(),
        c="blue",
    )
    if dbg_path is not None:
        plt.savefig(dbg_path)
    if show:
        plt.show()
    plt.close()


def plot_2d_tensor(data: Tensor, dbg_path: Optional[str] = None, show=False):
    plt.scatter(
        data[:, 0].cpu().detach().numpy(),
        data[:, 1].cpu().detach().numpy(),
    )
    if dbg_path is not None:
        plt.savefig(dbg_path)
    if show:
        plt.show()
    plt.close()


def visualize_otflow(
    flow: OTFlow, P_data, Q_data, dbg_path: Optional[str] = None, show=False
):
    """Plots both Q_data and Q_hat = flow(P_data)"""
    blocks = flow.get_blocks()
    prev_P = P_data
    for index, block in enumerate(blocks):
        is_reverse = index >= len(flow.jko1)
        prev_P, _, _ = block(prev_P, reverse=is_reverse)
        plot_two_distributions(prev_P, Q_data, dbg_path=dbg_path, show=show)


def visualize_otflow_trajectory(flow: OTFlow, P_data, dbg_path: str, reverse=False):
    """Generate gif of trajectory using flow model from initial P_data. Saves gif to dbg_path."""
    if P_data.shape[-1] != 2:
        raise NotImplementedError

    frames = []
    blocks: list[ODEFuncBlock] = list(flow.get_blocks(reverse=reverse))
    for block in blocks:
        block.device = flow.device

    prev_P = P_data
    for index, block in enumerate(blocks):
        is_reverse = index >= len(flow.jko1)
        xT, _, _ = block(prev_P, reverse=is_reverse, full_traj=True)
        prev_P = xT[-1]

        # have to make a copy i think, shape (T,B, 2)
        xs = xT[1:, :, :]  # cut off first in time step
        for t in range(xs.shape[0]):
            frames.append(xs[t].detach().cpu().numpy())

    for end_frames in range(20):
        frames.append(frames[-1])

    fig, ax = plt.subplots(figsize=(5, 5))
    x0 = frames[0]
    s, fps = 8, 12
    scat = ax.scatter(x0[0], x0[1], s=s, alpha=0.9)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("Particle flow")

    def init():
        scat.set_offsets(frames[0])
        return (scat,)

    def update(i):
        scat.set_offsets(frames[i])
        return (scat,)

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(frames), interval=1000 / fps, blit=True
    )
    writer = PillowWriter(fps=fps)
    anim.save(dbg_path, writer=writer)
    plt.close(fig)


def visualize_DRE(classifier: nn.Module):
    # same as gen data from image bounds
    x = torch.linspace(-4, 4, 100)
    y = torch.linspace(-4, 4, 100)
    x, y = torch.meshgrid([x, y])

    xx = torch.flatten(x)
    yy = torch.flatten(y)
    all_points = torch.stack([xx, yy], dim=1)

    ret = classifier(all_points)
    ret = ret.squeeze().detach().cpu().numpy()
    ret = ret.reshape(-1, 100)

    fig, ax = plt.subplots(figsize=(5, 5))

    # sns.heatmap(ret, cmap="coolwarm")
    mesh = ax.pcolormesh(x, y, ret, shading="auto", cmap="coolwarm")
    ax.set_title("log (p/q)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Value")
    plt.savefig("assets/DRE_estimate.png")
    plt.show()
