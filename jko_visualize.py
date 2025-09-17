import torch
from jko_iflow import (
    JKO,
    ODEFuncBlock,
    NN,
)
import matplotlib.pyplot as plt
import numpy as np


"""Random testing python file that loades in model from save_path."""


def main():
    save_path = "train_1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdict = torch.load(f"chkpt/{save_path}.pth", map_location=device)
    args = sdict["args"]
    h_ks = sdict["h_ks"]

    flow = JKO(
        [
            ODEFuncBlock(
                h_ks[idx],
                NN(device),
                args["sigma_0"] / np.sqrt(args["d"]),
                ode_solver="rk4",
                device=device,
            )
            for idx in range(args["num_blocks"])
        ]
    ).to(device)
    flow.load_state_dict(sdict["model"])

    num_points = 10000

    points = torch.randn((num_points, 2)).to(device)
    with torch.no_grad():
        for idx, block in reversed(list(enumerate(flow.blocks))):
            p_z = points

            x = p_z[:, 0].cpu().detach().numpy()
            y = p_z[:, 1].cpu().detach().numpy()
            # sns.kdeplot(x=x, y=y, fill=True, cmap="mako", thresh=0.001, bw_adjust=0.3)
            plt.show()

            plt.scatter(
                x=x,
                y=y,
                s=5,
                alpha=0.3,
            )
            plt.title(f"block idx: {idx}")
            plt.savefig(f"plot_{idx}.png")
            plt.close()
            p_x, _, _ = block(points, reverse=True)

            points = p_x


if __name__ == "__main__":
    main()
