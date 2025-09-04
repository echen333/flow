import torch
from models import OTFlow, BasicClassifier, JKO, ODEFunction
from data_gen import get_gaussian
from torch.utils.data import IterableDataset, DataLoader
import yaml
from jko_iflow import train_flow, ResamplingDataset, get_JKO
import numpy as np
from operator import itemgetter
from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
import wandb


def init_and_train_jkos(
    P_data: IterableDataset, Q_data: IterableDataset, device: str, *, args
) -> tuple[JKO, JKO]:
    """Initialize and generate warmstart for two JKO-scheme models that translate from P -> N(O,I) -> Q

    Args:
        P_data (IterableDataset): Dataset to sample from P
        Q_data (IterableDataset): Dataset to sample from Q
        device (str): torch device
        args (dict): Must contain parameters h_0, h_max, rho, num_blocks, sigma_0, and d for jko parameters

    Returns:
        tuple[JKO, JKO]: Returns trained models jko1, jko2
    """
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


def update_classifier(classifier: nn.Module, clf_optim, X, y):
    clf_optim.zero_grad()

    logits = classifier(X).squeeze(-1)
    clf_loss = F.binary_cross_entropy_with_logits(logits, y)
    clf_loss.backward()
    clf_optim.step()


def train_OT_Flow(
    model: OTFlow,
    classifier1: nn.Module,
    classifier2: nn.Module,
    P_data: IterableDataset,
    Q_data: IterableDataset,
    *,
    device: str = "cpu",
    tot_iters: int = 10,
    lr: float = 3e-4,
    clf_lr: float = 3e-4,
    E: int = 20,
    E_0: int = 20,
    E_in: int = 5,
    gamma: float = 1,
) -> None:
    ot_optim = torch.optim.AdamW(model.parameters(), lr=lr)
    clf1_optim = torch.optim.AdamW(classifier1.parameters(), lr=clf_lr)
    clf2_optim = torch.optim.AdamW(classifier2.parameters(), lr=clf_lr)

    # so P_data and Q_data just gives samples as in a minibatch
    # and we update all params after
    for iter_index, (P_sample, Q_sample) in enumerate(zip(P_data, Q_data)):
        N, D = P_sample.shape
        M, D2 = Q_sample.shape
        if D != D2:
            raise ValueError("Samples from P and Q must have the same dimension.")

        Q_hat = model(P_sample, t=0.0).detach()
        X = torch.concat([Q_hat, Q_sample])
        y = torch.concat([torch.zeros(N, device=device), torch.ones(M, device=device)])
        if iter_index == 0:
            for clf_epoch in range(E_0):
                print(f"starting epoch {clf_epoch}")
                update_classifier(classifier1, clf1_optim, X, y)

        for epoch in range(E):
            print(f"starting epoch {epoch}")
            ot_optim.zero_grad()

            classifier1.eval()
            KL_loss = -classifier1(Q_hat).mean()
            W2_loss = model.get_W2_loss(P_sample, reverse=False)
            loss = KL_loss + gamma * W2_loss
            # breakpoint()
            loss.backward()
            ot_optim.step()

            classifier1.train()

            # fit classifier 1 on new OT model
            for _ in range(E_in):
                Q_hat = model(P_sample).detach()
                X = torch.concat([Q_hat, Q_sample])
                y = torch.concat(
                    [torch.zeros(N, device=device), torch.ones(M, device=device)]
                )
                update_classifier(classifier1, clf1_optim, X, y)

        # now do the loss in reverse to go from Q -> P
        wandb_log = {}
        wandb.log(wandb_log)


def main():
    with open("config_ot_gaussians.yaml") as f:
        args = yaml.safe_load(f)
    run = wandb.init(entity="eddys", project="flow_basics", config=args)
    torch.random.manual_seed(42)

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

    otflow = OTFlow(jko1=jko1, jko2=jko2)
    tot_iters, lr, clf_lr = itemgetter("tot_iters", "lr", "clf_lr")(args["ot_train"])
    tot_iters, lr, clf_lr = int(tot_iters), float(lr), float(clf_lr)
    P_data.num_epochs = tot_iters
    Q_data.num_epochs = tot_iters

    dim = args["jko"]["d"]
    train_OT_Flow(
        otflow,
        BasicClassifier(dim),
        BasicClassifier(dim),
        P_data=P_data,
        Q_data=Q_data,
        tot_iters=tot_iters,
        lr=lr,
        clf_lr=clf_lr,
    )


if __name__ == "__main__":
    main()
