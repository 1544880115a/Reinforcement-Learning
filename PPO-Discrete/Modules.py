from torch import nn
from torch.nn import functional as F

class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 16), nn.ReLU(), 
            nn.Linear(16, 1))
    
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 16), nn.ReLU(), 
            nn.Linear(16, n_act))
        
    def forward(self, x):
        out = self.net(x)
        return F.softmax(out)