import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim=256, hidden_dim=1024, num_layers=2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z
