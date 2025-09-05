import torch
from torch import Tensor
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset, IterableDataset
import os
from visualize_data import plot_2d_tensor
import time
from operator import itemgetter
from data_gen import sample_points_from_image
from models import JKO, ODEFuncBlock, NN


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

    prev_points: Tensor = train_dataset  # (N,D)

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
            "device": device,
        }
        sdict = {
            "model": flow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args,
        }
        os.makedirs("chkpt", exist_ok=True)
        torch.save(sdict, f"chkpt/{save_path}_{block_idx}.pth")
        plot_2d_tensor(prev_points, f"dbg/block_{block_idx}.png")


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


# TODO: Remove
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
        num_epochs=train_args["epochs_per_block"],
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
        num_epochs=train_args["epochs_per_block"],
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
