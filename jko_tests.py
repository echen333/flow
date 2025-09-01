import torch
import torch.nn as nn
import os
import yaml
from flow import JKO, ODEFuncBlock, NN, sample_points_from_image, reparameterize_trajectory
import matplotlib.pyplot as plt
import numpy as np

def test_forward():
    pass

def main():
    save_path = "train_1"
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    sdict = torch.load(f"chkpt/{save_path}.pth")
    args = sdict["args"]
    h_ks = sdict["h_ks"]
    # h_ks =[np.float64(0.08704150759150331), np.float64(0.1831668147369398), np.float64(0.5342127305485939), np.float64(1.4043969599112267), np.float64(2.8153883490293747), np.float64(5.0), np.float64(5.0), np.float64(5.0)]

    flow = JKO(
        [
            ODEFuncBlock(
                h_ks[idx], NN(device), args["sigma_0"] / args["d"], ode_solver="rk4", device=device
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
            
            plt.scatter(x=p_z[:, 0].cpu().detach().numpy(), y=p_z[:, 1].cpu().detach().numpy(), s=5, alpha=0.3)
            plt.title(f"block idx: {idx}")
            plt.savefig(f'plot_{idx}.png')
            plt.close()
            p_x, _, _ = block(points, reverse=True)

            points = p_x


if __name__ == "__main__":
    main()
