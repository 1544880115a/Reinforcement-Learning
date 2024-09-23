import numpy as np

class Agent():
    def __init__(self, n_obs, n_act, gamma, epsilon):
        self.Q = np.zeros((n_obs, n_act))
        self.n_act = n_act

        self.gamma = gamma
        self.epsilon = epsilon

    def predict(self, obs):
        Q_list = self.Q[obs, :]
        return np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
    
    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_act)
        else:
            return self.predict(obs)
        
    def learn(self, obs_list, action_list, reward_list):
        G = 0 #记录该条episode的return
        for i in reversed(range(len(obs_list))): #从后往前进行计算
            #获取每一步的obs
            obs = obs_list[i]
            #获取每一步的action
            action = action_list[i]
            #获取每一步的reward
            reward = reward_list[i]
            #计算当前时刻的q值 = reward + γ * 下一时刻的q值
            G = reward + self.gamma * G
            #用q值对当前[s, a]对的q值进行更新
            self.Q[obs, action] = G