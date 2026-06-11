import torch
import torch.nn as nn
ALPHA_INIT = 0.5

class InversePINN(nn.Module):
    def __init__(self, T=2.0, width=64, depth=4):
        super().__init__()

        self.T = T

        layers = []

        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))
        self.alpha = torch.nn.Parameter(torch.tensor(ALPHA_INIT, dtype=torch.float32, device='cpu'))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        x_norm = 2.0 * x - 1.0
        t_norm = 2.0 * t / self.T - 1.0

        z = torch.cat([x_norm, t_norm], dim=1)
        u = self.net(z)

        return u
