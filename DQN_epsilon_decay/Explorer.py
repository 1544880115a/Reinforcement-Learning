import numpy as np

class Explorer():
    def __init__(self, n_act, epsilon, decay):
        self.n_act = n_act
        self.epsilon = epsilon
        self.decay = decay

    def act(self, predict_method, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = predict_method(obs)
        self.epsilon = max(0.01, self.epsilon - self.decay)
        return action