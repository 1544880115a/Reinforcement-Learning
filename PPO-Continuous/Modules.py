from torch import nn
from torch.nn import functional as F
import torch

class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 16), nn.ReLU(), 
            nn.Linear(16, 1))
    
    def forward(self, x):
        return self.net(x)

#输出连续动作的高斯分布的均值和标准差    
class Actor(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.fc = nn.Linear(n_obs, 16)
        self.fc_u = nn.Linear(16, n_act)
        self.fc_std = nn.Linear(16, n_act)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        u = self.fc_u(x)
        u = 2 * torch.tanh(u) #将均值控制在[-1, 1]
        std = self.fc_std(x)
        std = F.softplus(std) #将标准差小于0的部分逼近0， 大于0的部分几乎不变

        return u, std