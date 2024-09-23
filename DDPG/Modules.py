import torch
from torch import nn

class Critic(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, 64), nn.ReLU(), #这里拼起来是因为此时aciton已不是离散值，
            nn.Linear(64, 64), nn.ReLU(), #无法通过[action]的方式访问Q值，
            nn.Linear(64, 1)) #只能将obs和action同时作为参数传入神经网络中
        
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.net(cat)
    
class Actor(nn.Module):
    def __init__(self, n_obs, n_act, action_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(), 
            nn.Linear(64, n_act))
        self.action_bound = action_bound
    
    def forward(self, x):
        out = self.net(x)
        out = torch.tanh(out) #将action值压缩在[-action_bound, action_bound]内
        return self.action_bound * out