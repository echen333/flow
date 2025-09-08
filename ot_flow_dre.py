import torch
import yaml
from models import OTFlow, BasicClassifier
import torch.nn as nn
from torch import Tensor
from data_gen import get_samples
from visualize_data import visualize_DRE
import torch.nn.functional as F


def train_DRE_classifier(
    classifier: nn.Module, ot_model: OTFlow, lr, P_data: Tensor, Q_data: Tensor
):
    device = classifier.device

    forward_data = ot_model(
        P_data, reverse=False, all_block_outputs=True, full_traj=False
    )
    backwards_data = ot_model(
        Q_data, reverse=True, all_block_outputs=True, full_traj=False
    )
    backwards_data = torch.flip(backwards_data, dims=[0])  # reverse across time
    assert forward_data.shape == backwards_data.shape

    num_epochs = 50
    batch_size = 512
    # breakpoint()
    L, B = forward_data.shape[0] - 1, forward_data.shape[1]

    # breakpoint()
    optimizer = torch.optim.Adam(classifier.parameters(), lr)
    for epoch in range(num_epochs):
        for start_idx in range(0, B, batch_size):
            loss = torch.tensor([0], device=device, dtype=torch.float32)
            for idx in range(L):
                end_idx = min(start_idx + batch_size, B)
                batch_len = end_idx - start_idx

                forward_neg = forward_data[idx][start_idx:end_idx]
                forward_pos = forward_data[idx + 1][start_idx:end_idx]
                backward_neg = backwards_data[idx][start_idx:end_idx]
                backward_pos = backwards_data[idx + 1][start_idx:end_idx]

                X = torch.concat([forward_neg, forward_pos, backward_neg, backward_pos])
                y = torch.concat(
                    [
                        torch.zeros((batch_len,), device=device),
                        torch.ones((batch_len,), device=device),
                        torch.zeros((batch_len,), device=device),
                        torch.ones((batch_len,), device=device),
                    ]
                )
                logits = classifier(X).squeeze(-1)
                idx_loss = F.binary_cross_entropy_with_logits(logits, y)
                # breakpoint()
                loss += idx_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"loss at epoch {epoch} is {loss.item(): .2f} avg_loss: {loss.item() / L: .3f}"
            )

    save_path = "chkpt/classifier.pt"
    save_obj = {
        "classifier": classifier,
        "clf_optim": optimizer.state_dict(),
        "ot_model": ot_model,
        "P_data": P_data,
        "Q_data": Q_data,
    }
    torch.save(save_obj, save_path)


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "chkpt/moons_to_checkerboard_ot_1.pt"
    saved_obj = torch.load(model_path, map_location=device, weights_only=False)
    model = saved_obj["ot_model"]
    model.device = device
    for block in model.get_blocks():
        block.device = device

    classifier = BasicClassifier(2, device=device)

    P_data = get_samples("moons", 2048, device=device, prefix=None)
    Q_data = get_samples(
        "image",
        2048,
        device=device,
        prefix="Q",
        Q_image_path="assets/img_checkerboard.png",
    )

    train_DRE_classifier(classifier, model, 0.001, P_data, Q_data)
    visualize_DRE(classifier)


if __name__ == "__main__":
    main()
