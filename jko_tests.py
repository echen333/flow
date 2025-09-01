import torch
import torch.nn as nn
import os
import yaml
from flow import JKO, ODEFuncBlock, NN, sample_points_from_image
import matplotlib.pyplot as plt


def main():
    with open("config_rose.yaml") as f:
        args = yaml.safe_load(f)
    test_args = args["test"]
    sdict = torch.load(f"chkpt/{test_args['save_path']}_9.pth")
    # breakpoint()
    train_args = sdict["args"]
    h_ks = [
        min(args["h_max"], args["h_0"] * (args["rho"] ** idx))
        for idx in range(args["num_blocks"])
    ]
    flow = JKO(
        [
            ODEFuncBlock(
                h_ks[idx], NN(), args["sigma_0"] / args["d"], ode_solver="dopri5"
            )
            for idx in range(train_args["num_blocks"])
        ]
    )
    flow.load_state_dict(sdict["model"])

    num_points = 5000
    # points = torch.randn((num_points, 2))
    # p_x, _ = flow(points, 0, None, True)
    # plt.scatter(x=p_x[:, 0].detach().numpy(), y=p_x[:, 1].detach().numpy())
    # plt.show()

    points = torch.randn((num_points, 2))
    # fig, axs = plt.subplots(3, 3)
    with torch.no_grad():
        for idx, block in reversed(list(enumerate(flow.blocks))):
            p_z = points
            plt.scatter(x=p_z[:, 0].detach().numpy(), y=p_z[:, 1].detach().numpy())
            plt.title(f"block idx: {idx}")
            plt.show()
            p_x, _ = block(points, reverse=False)

            points = p_x

    # points2 = sample_points_from_image(args["image_path"], 500)
    # zz, _ = flow(points2, 0)
    # breakpoint()
    # plt.scatter(x=zz[:, 0].detach().numpy(), y=zz[:, 1].detach().numpy())
    # plt.show()


if __name__ == "__main__":
    main()
