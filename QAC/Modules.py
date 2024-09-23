from torch import nn
from torch.nn import functional as F

class Actor(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 100), nn.ReLU(), 
            nn.Linear(100, 100), nn.ReLU(), 
            nn.Linear(100, n_act))
    
    def forward(self, x):
        out = self.net(x)
        return F.softmax(out)
    
class Critic(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 100), nn.ReLU(), 
            nn.Linear(100, 100), nn.ReLU(), 
            nn.Linear(100, n_act))
        
    def forward(self, x):
        return self.net(x)