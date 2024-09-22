import numpy as np
import torch

class Agent():
    def __init__(self, q_func, optimizer, replay_buffer, minibatch_size, 
                 replay_start_size, n_act, gamma, e):
        self.global_steps = 0
        self.q_func = q_func
        self.optimizer = optimizer
        self.loss = torch.nn.MSELoss()

        self.rb = replay_buffer
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size

        self.n_act = n_act
        self.gamma = gamma
        self.e = e

    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.q_func(obs)
        return int(torch.argmax(Q_list).detach().numpy())
    
    def act(self, obs):
        if np.random.uniform(0, 1) < self.e:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action
    
    def learn_batch(self):
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.rb.sample(self.minibatch_size)
        predict_q1 = self.q_func(batch_obs)
        predict_q2 = predict_q1[torch.arange(len(predict_q1)), batch_action]
        target_q = batch_reward + (1 - batch_done) * self.gamma * self.q_func(batch_next_obs).max(1)[0]

        self.optimizer.zero_grad()
        l = self.loss(predict_q2, target_q)
        l.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_steps += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) >= self.replay_start_size and self.global_steps % self.rb.num_steps == 0:
            self.learn_batch()