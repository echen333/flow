import torch
from visualize_data import visualize_otflow_trajectory
from data_gen import get_samples


"""Generate the gif of moons < --- > checkerboard using model from save_path."""


def main():
    save_path = "chkpt/moons_to_checkerboard_ot_1.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdict = torch.load(save_path, map_location=device, weights_only=False)
    flow_model = sdict["ot_model"]
    flow_model.device = device

    P_data = get_samples("moons", 2048, device=device, prefix=None)
    visualize_otflow_trajectory(flow_model, P_data, "assets/moons_traj.gif")
    Q_data = get_samples(
        "image",
        2048,
        device=device,
        prefix="Q",
        Q_image_path="assets/img_checkerboard.png",
    )
    visualize_otflow_trajectory(
        flow_model, Q_data, "assets/moons_traj2.gif", reverse=True
    )


if __name__ == "__main__":
    main()
