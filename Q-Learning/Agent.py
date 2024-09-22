import numpy as np

class Agent():
    def __init__(self, n_obs, n_act, lr, gamma, e):
        self.Q = np.zeros((n_obs, n_act))
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma
        self.e = e

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        return np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
    
    def act(self, obs):
        if np.random.uniform(0, 1) < self.e:
            return np.random.choice(self.n_act)
        else:
            return self.predict(obs)
        
    def learn(self, obs, action, reward, next_obs, done):
        cur_q = self.Q[obs, action]
        target_q = reward + (1 - float(done)) * self.gamma * self.Q[next_obs, :].max()

        self.Q[obs, action] -= self.lr * (cur_q - target_q)