import torch
import torch.nn as nn
import os
import yaml
from flow import JKO, Block, NN


def main():
    with open("config_rose.yaml") as f:
        args = yaml.safe_load(f)
    test_args = args["test"]
    sdict = torch.load(f"chkpt/{test_args['save_path']}_0.pth")
    # breakpoint()
    train_args = sdict["args"]
    h_ks = [
        min(args["h_max"], args["h_0"] * (args["rho"] ** idx))
        for idx in range(args["num_blocks"])
    ]
    flow = JKO(
        [
            Block(h_ks[idx], NN(), args["sigma_0"] / args["d"])
            for idx in range(train_args["num_blocks"])
        ]
    )
    flow.load_state_dict(sdict["model"])


if __name__ == "__main__":
    main()
