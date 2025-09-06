import torch
from models import OTFlow, BasicClassifier, JKO
from data_gen import get_samples
from torch.utils.data import DataLoader, TensorDataset
import yaml
from jko_iflow import train_flow, get_JKO
from operator import itemgetter
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
import os
import time
from visualize_data import plot_two_distributions, visualize_otflow
from typing import Optional


def init_and_train_jkos(
    P_data: Tensor, Q_data: Tensor, device: str, *, args, ot_save_path
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

    lr, batch_size, clip_grad, save_path, epochs_per_block = itemgetter(
        "lr", "batch_size", "clip_grad", "save_path", "epochs_per_block"
    )(args["jko_train"])
    train_flow(
        jko1,
        P_data,
        num_epochs=epochs_per_block,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )
    train_flow(
        jko2,
        Q_data,
        num_epochs=epochs_per_block,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
    )

    save_obj = {"jko1": jko1, "jko2": jko2}
    torch.save(save_obj, ot_save_path)
    return jko1, jko2


def update_classifier(
    classifier: nn.Module, clf_optim, D_hat: Tensor, D_sample: Tensor
) -> Tensor:
    N, M = D_hat.shape[0], D_sample.shape[0]
    device = classifier.device
    X = torch.concat([D_hat, D_sample])
    y = torch.concat([torch.zeros(N, device=device), torch.ones(M, device=device)])
    clf_optim.zero_grad()

    logits = classifier(X).squeeze(-1)
    clf_loss = F.binary_cross_entropy_with_logits(logits, y)
    clf_loss.backward()
    clf_optim.step()
    return clf_loss


def upload_artifact(path: str):
    """just scp model"""
    # run = wandb.init(entity="eddys", project="flow_basics", job_type="add_model")
    # artifact = wandb.Artifact(name=f"path_{path}", type="model")
    # pass


def train_OT_Flow(
    model: OTFlow,
    classifier1: nn.Module,
    classifier2: nn.Module,
    P_data: Tensor,
    Q_data: Tensor,
    *,
    batch_size: int = 500,
    Tot: int = 2,
    lr: float = 3e-4,
    clf_lr: float = 3e-4,
    E: int = 50,
    E_0: int = 300,
    E_in: int = 4,
    gamma: float = 0.5,
    optim_cls=torch.optim.AdamW,
    save_path: Optional[str] = None,
    args: Optional[dict] = None
) -> None:
    """Train an OTFlow model as outlined in Algo 1 
    in 'Computing high-dimensional optimal transport by flow neural networks.'

    Args:
        model (OTFlow): warm-start flow model
        classifier1 (nn.Module): Classifier c_1
        classifier2 (nn.Module): Classifier \tilde{c_0}
        P_data (Tensor): Samples from distribution P
        Q_data (Tensor): Samples from distribution Q
        batch_size (int, optional): Batch size. Defaults to 500.
        device (str, optional): device. Defaults to "cpu".
        Tot (int, optional): Described in Algo 1. Defaults to 2.
        lr (float, optional): Learning Rate. Defaults to 3e-4.
        clf_lr (float, optional): Learning rate for classifier1 and 2. Defaults to 3e-4.
        E (int, optional): Described in Algo 1. Defaults to 50.
        E_0 (int, optional): Described in Algo 1. Defaults to 300.
        E_in (int, optional): Described in Algo 1. Defaults to 4.
        gamma (float, optional): Weight of L_T in Eqn 4. Defaults to 0.5.
        optim_cls (Optimizer_cls., optional): Optimizer class. Defaults to torch.optim.AdamW.
        save_path (str, optional): Model save path
        args (dict, optional): Args

    Raises:
        ValueError: _description_
    """
    ot_optim = optim_cls(model.parameters(), lr=lr)
    clf1_optim = optim_cls(classifier1.parameters(), lr=clf_lr)
    clf2_optim = optim_cls(classifier2.parameters(), lr=clf_lr)

    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    global_step = 0

    if P_data.shape[1] != Q_data.shape[1]:
        raise ValueError("Samples from P and Q must have the same dimension.")

    print(f"Begin training of OTFlow")
    for iter_index in range(Tot):
        # TODO: should not be using DataLoader because paying for CPU syncs
        P_data_loader = DataLoader(TensorDataset(P_data), batch_size, shuffle=True)
        Q_data_loader = DataLoader(TensorDataset(Q_data), batch_size, shuffle=True)
        assert len(P_data) == len(Q_data), "hmmm not necessarily right but placeholder"

        if iter_index == 0:
            start_time = time.time()
            for clf_step in range(E_0):
                # Tensor Dataset returns a tuple so we have to take in as (P_sample, )
                for (P_sample,), (Q_sample,) in zip(P_data_loader, Q_data_loader):
                    Q_hat = model(P_sample, t=0.0).detach()
                    clf_loss: Tensor = update_classifier(classifier1, clf1_optim, Q_hat, Q_sample)
                    wandb.log({"init_clf_step": clf_step, "init_clf_loss": clf_loss.item()})
            print(f"initialization of clf0 took {time.time() - start_time}s")

        for epoch in range(E):
            first_data = True

            print(f"starting epoch {epoch}")
            for (P_sample,), (Q_sample,) in zip(P_data_loader, Q_data_loader):
                ot_optim.zero_grad()

                Q_hat = model(P_sample, t=0.0)
                classifier1.eval()
                KL_loss = -classifier1(Q_hat).mean()
                classifier1.train()
                W2_loss = model.get_W2_loss(P_sample, reverse=False)
                loss = KL_loss + gamma * W2_loss
                loss.backward()
                ot_optim.step()

                if first_data and epoch % 5 == 0:
                    first_data = False
                    # lets plot Q_hat and Q
                    plot_two_distributions(
                        Q_hat, Q_sample, dbg_path=f"dbg/ot_flow_fwd_epoch_{iter_index}{1}_{epoch}.png"
                    )

                # fit classifier 1 on new OT model
                clf_loss = None
                model.eval()
                for _ in range(E_in):
                    Q_hat = model(P_sample).detach()
                    clf_loss = update_classifier(
                        classifier1, clf1_optim, Q_hat, Q_sample
                    )
                model.train()

                wandb_log = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "iter_index": iter_index,
                    "loss": loss.item(),
                    "KL_loss": KL_loss.item(),
                    "gamma_W2_loss": gamma * W2_loss.item(),
                    "W2_loss": W2_loss.item(),
                    "clf_losses": clf_loss.item(),
                    "grad_norm": next(model.parameters()).grad.data.norm(2).item(),
                }
                global_step += 1
                wandb.log(wandb_log)

        if iter_index == 0:
            start_time = time.time()
            for clf_step in range(E_0):
                # Tensor Dataset returns a tuple so we have to take in as (P_sample, )
                for (P_sample,), (Q_sample,) in zip(P_data_loader, Q_data_loader):
                    P_hat = model(Q_sample, t=0.0, reverse=True).detach()
                    clf_loss: Tensor = update_classifier(classifier2, clf2_optim, P_hat, P_sample)
                    wandb.log({"init_clf_step": clf_step, "init_clf_loss": clf_loss.item()})
            print(f"initialization of clf0 took {time.time() - start_time}s")

        for epoch in range(E):
            first_data = True

            print(f"starting epoch {epoch}")
            for (P_sample,), (Q_sample,) in zip(P_data_loader, Q_data_loader):
                ot_optim.zero_grad()

                P_hat = model(Q_sample, t=0.0, reverse=True)
                classifier2.eval()
                KL_loss = -classifier2(P_hat).mean()
                classifier2.train()
                W2_loss = model.get_W2_loss(Q_sample, reverse=True)
                loss = KL_loss + gamma * W2_loss
                loss.backward()
                ot_optim.step()

                if first_data and epoch % 5 == 0:
                    first_data = False
                    # lets plot P_hat and P
                    plot_two_distributions(
                        P_hat, P_sample, dbg_path=f"dbg/ot_flow_fwd_epoch_{iter_index}{2}_{epoch}.png"
                    )

                # fit classifier 2 on new OT model
                clf_loss = None
                model.eval()
                for _ in range(E_in):
                    P_hat = model(Q_sample, reverse=True).detach()
                    clf_loss = update_classifier(
                        classifier2, clf2_optim, P_hat, P_sample
                    )
                model.train()

                wandb_log = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "iter_index": iter_index,
                    "loss": loss.item(),
                    "KL_loss": KL_loss.item(),
                    "gamma_W2_loss": gamma * W2_loss.item(),
                    "W2_loss": W2_loss.item(),
                    "clf_losses": clf_loss.item(),
                    "grad_norm": next(model.parameters()).grad.data.norm(2).item(),
                }
                global_step += 1
                wandb.log(wandb_log)

        save_object = {
            "ot_model": model,
            "clf1": classifier1,
            "clf2": classifier2,
            "args": args,
            "ot_optim": ot_optim.state_dict(),
            "clf1_optim": clf1_optim.state_dict(),
            "clf2_optim": clf2_optim.state_dict(),
        }
        if save_path is not None:
            print(f"saving object to {save_path}_{iter_index}.pt")
            torch.save(save_object, f"{save_path}_{iter_index}.pt")


def main():
    with open("config_ot_gaussians.yaml") as f:
        args = yaml.safe_load(f)
    print("args:", args)
    run = wandb.init(entity="eddys", project="flow_basics", config=args)
    torch.random.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get data samples for P and Q
    data_args = args["data"]
    P_data = get_samples(
        data_args["P_distribution"],
        prefix="P",
        num_samples=args["N"],
        device=device,
        **data_args,
    )
    Q_data = get_samples(
        data_args["Q_distribution"],
        prefix="Q",
        num_samples=args["M"],
        device=device,
        **data_args,
    )

    save_path = f"chkpt/{args['save_path']}.pt"
    os.makedirs("dbg/", exist_ok=True)
    os.makedirs("chkpt/", exist_ok=True)
    plot_two_distributions(P_data, Q_data, dbg_path="dbg/tmp_P_and_Q.png")

    # Initialize (and warm-start) our two JKO models. 
    if not args["load_JKO"] or not os.path.exists(save_path):
        jko1, jko2 = init_and_train_jkos(P_data, Q_data, device=device, args=args, ot_save_path=save_path)
    else:
        save_obj = torch.load(save_path, weights_only=False)
        jko1 = save_obj["jko1"]
        jko2 = save_obj["jko2"]

    # Concat the two JKO models to make an Optimal Transport (OT) model
    otflow = OTFlow(jko1=jko1, jko2=jko2)
    lr, clf_lr, gamma = map(
        float, itemgetter("lr", "clf_lr", "gamma")(args["ot_train"])
    )
    batch_size, Tot, E, E_0, E_in = map(
        int, itemgetter("batch_size", "Tot", "E", "E_0", "E_in")(args["ot_train"])
    )
    visualize_otflow(otflow, P_data[:batch_size], Q_data[:batch_size], dbg_path="dbg/after_jko_warm_start.png")

    # Initialize classifiers used in Algo 1 and train our OT model
    dim = args["jko"]["d"]
    clf1 = BasicClassifier(dim, device)
    clf2 = BasicClassifier(dim, device)
    train_OT_Flow(
        otflow,
        clf1,
        clf2,
        P_data=P_data,
        Q_data=Q_data,
        batch_size=batch_size,
        Tot=Tot,
        lr=lr,
        clf_lr=clf_lr,
        E=E,
        E_0=E_0,
        E_in=E_in,
        gamma=gamma,
        args=args,
        save_path=f"{save_path}_ot"
    )


if __name__ == "__main__":
    main()
