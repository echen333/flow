import torch
import matplotlib.pyplot as plt
from models import OTFlow
from typing import Optional
from torch import Tensor


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

def plot_2d_tensor(data: Tensor, dbg_path: Optional[str]=None, show=False):
    plt.scatter(
        data[:, 0].cpu().detach().numpy(),
        data[:, 1].cpu().detach().numpy(),
    )
    if dbg_path is not None:
        plt.savefig(dbg_path)
    if show:
        plt.show()
    plt.close()

def visualize_otflow(flow: OTFlow, P_data, Q_data, dbg_path: Optional[str]=None, show=False):
    blocks = flow.get_blocks()
    prev_P = P_data
    for index, block in enumerate(blocks):
        is_reverse = index >= len(flow.jko1)
        prev_P, _, _ = block(prev_P, reverse=is_reverse)
        plot_two_distributions(prev_P, Q_data, dbg_path=dbg_path, show=show)
