import torch
from torch import Tensor
from sklearn.datasets import make_moons
import numpy as np
from PIL import Image


def get_gaussian(mu: Tensor, sigma: float, num_samples: int, device="cpu"):
    x = torch.randn(num_samples, *mu.shape) * sigma + mu
    return x.to(device)


def gen_moons(num_samples: int, device="cpu", seed=42):
    xraw, _ = make_moons(noise=0.05, n_samples=num_samples, random_state=seed)
    # Scale to same domain as checkerboard
    mean = xraw.mean(axis=0)
    std = xraw.std(axis=0) / np.array([np.sqrt(4), np.sqrt(5)])
    xraw = (xraw - mean) / std
    xraw = torch.from_numpy(xraw).float().to(device)
    return xraw


def sample_points_from_image(image_path: str, num_points=10000):
    """
    Data generation of distribution from image, fit to [-4,4].
    Taken from https://github.com/hamrel-cxu/JKO-iFlow
    """
    image = Image.open(image_path)
    # not sure where transforms but grayscale image
    image_mask = np.array(image.rotate(180).transpose(0).convert("L"))

    w, h = image.size
    x = np.linspace(-4, 4, w)
    y = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))
    means = np.concatenate([xx, yy], 1)

    img = image_mask.max() - image_mask
    probs = img.reshape(-1) / img.sum()
    std = np.array([8 / w / 2, 8 / h / 2])
    inds = np.random.choice(int(probs.shape[0]), num_points, p=probs)
    m = means[inds]
    samples = np.random.randn(*m.shape) * std + m
    return torch.tensor(samples, dtype=torch.float32)


def get_samples(
    distribution: str, num_samples: int, device: str, prefix: str, **kwargs
):
    if distribution == "moons":
        return gen_moons(
            num_samples,
            device=device,
        )
    if distribution == "gaussian":
        mu, sigma = Tensor(kwargs[f"{prefix}_mu"]), float(kwargs[f"{prefix}_sigma"])
        return get_gaussian(mu, sigma, num_samples, device)
    if distribution == "image":
        return sample_points_from_image(kwargs[f"{prefix}_image_path"], num_samples).to(device)
    raise ValueError(f"Distribution {distribution} is not valid")
