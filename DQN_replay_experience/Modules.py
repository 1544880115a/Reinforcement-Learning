import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 100), nn.ReLU(), 
            nn.Linear(100, 100), nn.ReLU(), 
            nn.Linear(100, n_act))
        
    def forward(self, x):
        return self.net(x)