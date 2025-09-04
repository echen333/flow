import torch
from models import MLP, OTFlow
from data_gen import get_gaussian
import yaml
from jko_iflow import JKO, NN, ODEFuncBlock, train_flow, ResamplingDataset, get_JKO
import numpy as np
from operator import itemgetter
from typing import Callable


def init_and_train_jkos(P_data, Q_data, device, *, args):
    h_0, h_max, rho, num_blocks, sigma_0, d = itemgetter(
        "h_0", "h_max", "rho", "num_blocks", "sigma_0", "d"
    )(args["jko"])
    jko1 = get_JKO(
        h_0=h_0,
        h_max=h_max,
        rho=rho,
        num_blocks=num_blocks,
        sigma_0=sigma_0,
        d=d,
        device=device,
    )
    jko2 = get_JKO(
        h_0=h_0,
        h_max=h_max,
        rho=rho,
        num_blocks=num_blocks,
        sigma_0=sigma_0,
        d=d,
        device=device,
    )

    lr, batch_size, clip_grad, save_path = itemgetter(
        "lr", "batch_size", "clip_grad", "save_path"
    )(args["jko_train"])
    train_flow(
        jko1,
        P_data,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )
    train_flow(
        jko2,
        Q_data,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )

    save_obj = {"jko1": jko1, "jko2": jko2}
    ot_save_path = "chkpt/ot_flow_jko1_2.pt"
    torch.save(save_obj, ot_save_path)
    return jko1, jko2


def train_classifier():
    pass


def train_OT_Flow(
    model: OTFlow,
    classifier1_ctor: Callable[[], MLP],
    classifier2_ctor: Callable[[], MLP],
    *,
    tot_iters: int,
    refinenment_epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for iter in range(tot_iters):
        classifier_1 = classifier1_ctor()
        clf1_optim = torch.optim.SGD(classifier_1.parameters(), clf_lr)
        points1, points2 = None, None
        train_classifier(classifier_1, points1, points2)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loss = classifier_1(points)
            loss.backward()
            optimizer.step()
            pass

        classifier_2 = classifier1_ctor()
        points1, points2 = None, None
        train_classifier(classifier_2, points1, points2)
    pass


def main():
    with open("config_ot_gaussians.yaml") as f:
        args = yaml.safe_load(f)
    net = MLP((5, 10, 1), "relu")
    torch.random.manual_seed(42)

    print(net)
    print(get_gaussian(torch.tensor([0, 1]), 2, 10))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    P_mu, P_sigma = torch.tensor([5, 5]), 1

    num_epochs = 100
    P_data = ResamplingDataset(
        lambda: get_gaussian(P_mu, P_sigma, args["N"], device), num_epochs, device
    )
    Q_mu, Q_sigma = torch.tensor([-5, 5]), 2
    Q_data = ResamplingDataset(
        lambda: get_gaussian(Q_mu, Q_sigma, args["M"], device), num_epochs, device
    )

    save_path = "chkpt/ot_flow_jko1_2.pt"
    if not args["load_JKO"]:
        jko1, jko2 = init_and_train_jkos(P_data, Q_data, device=device, args=args)
    else:
        save_obj = torch.load(save_path, weights_only=False)
        jko1 = save_obj["jko1"]
        jko2 = save_obj["jko2"]

    otflow = OTFlow(jko1, jko2)


if __name__ == "__main__":
    main()
