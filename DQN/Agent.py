import numpy as np
import torch

class Agent():
    def __init__(self, q_func, optimizer, n_act, gamma, e):
        self.q_func = q_func
        self.optimizer = optimizer
        self.loss = torch.nn.MSELoss()

        self.n_act = n_act
        self.gamma = gamma
        self.e = e

    def predict(self, obs):
        Q_list = self.q_func(obs)
        return int(torch.argmax(Q_list).detach().numpy())
    
    def act(self, obs):
        if np.random.uniform(0, 1) < self.e:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action
    
    def learn(self, obs, action, reward, next_obs, done):
        predict_q = self.q_func(obs)[action]
        target_q = reward + (1 - float(done)) * self.gamma * self.q_func(next_obs).max()

        self.optimizer.zero_grad()
        l = self.loss(predict_q, target_q)
        l.backward()
        self.optimizer.step()