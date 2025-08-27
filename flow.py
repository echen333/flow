import torch
import torch.nn as nn
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x, t):
        x = self.act1(self.fc1(torch.cat([x, t], dim=1)))
        x = self.fc2(x)
        return x

class Hutchinson(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.epsilon = nn.Parameter(torch.randn(1))

    def forward(self, x, t):
        return self.epsilon * self.model(x, t)
class Block(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.ode_solver = "rk4"
        self.h_steps = 3
        self.hutchinson = Hutchinson(self.model)

    def forward(self, x, t):
        if self.ode_solver == "rk4":
            h = 1 / self.h_steps
            for i in range(self.h_steps):
                k1 = self.model(x, t)
                k2 = self.model(x + h * k1 / 2, t + h / 2)
                k3 = self.model(x + h * k2 / 2, t + h / 2)
                k4 = self.model(x + h * k3, t + h)
                x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # TODO: use hutchinson trace estimator to compute div_f
        jac = torch.autograd.functional.jacobian(self.hutchinson, x)
        div_f = torch.trace(jac)
        div_f = div_f.mean()
        
        return x, div_f

class JKO(nn.Module):
    def __init__(self, blocks: list[Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.block_length = len(blocks)
        
    def forward(self, x, t, block_idx = None):
        if block_idx is None:
            for block in self.blocks:
                x, div_f = block(x, t)
        else:
            x, div_f = self.blocks[block_idx](x, t)
        return x, div_f

def sample_points_from_image(image_path, num_points=100000):
    """
    Sample points from a black and white image.
    Sample points from the black part of the image.
    Returns a list of coordinates of the sampled points.
    """
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image = image.convert("L")
    
    points = []
    for i in range(num_points):
        x = random.randint(0, 511)
        y = random.randint(0, 511)  
        if image.getpixel((x, y)) < 100:
            points.append((x, y))
    return np.array(points)

def train_flow(flow: JKO, points: np.ndarray, num_steps=1000):
    """ train block by block """
    prev_points = points
    t = torch.linspace(0, 1, flow.block_length)
    for block_idx in range(flow.block_length):
        block = flow.blocks[block_idx]
        optimizer = torch.optim.Adam(block.parameters(), lr=0.01)
        for step in range(num_steps):
            optimizer.zero_grad()
            breakpoint()
            out, div_f = flow(prev_points, t, block_idx)
            
            V_loss = np.linalg.norm(out, axis=1).mean()
            delta = out - prev_points
            W_loss = np.linalg.norm(delta, axis=1).mean()
            loss = V_loss + div_f + W_loss
            
            loss.backward()
            optimizer.step()
            print(f"Block {block_idx} loss: {loss.item()}")
        prev_points = flow(prev_points, t, block_idx)

def main():
    image_path = "img_rose.png"
    points = sample_points_from_image(image_path)
    points = torch.from_numpy(points).float()
    
    flow = JKO([Block(NN()) for _ in range(10)])
    train_flow(flow, points)

if __name__ == "__main__":
    main()
