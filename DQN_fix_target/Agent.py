import numpy as np
import torch
import copy

class Agent():
    def __init__(self, q_func, optimizer, replay_buffer, minibatch_size, 
                 replay_start_size, target_update_interval, n_act, gamma, e):
        self.global_steps = 0
        self.pred_func = q_func
        self.target_func = copy.deepcopy(q_func)
        self.optimizer = optimizer
        self.loss = torch.nn.MSELoss()

        self.rb = replay_buffer
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size

        self.target_update_interval = target_update_interval
        self.n_act = n_act
        self.gamma = gamma
        self.e = e

    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.pred_func(obs)
        return int(torch.argmax(Q_list).detach().numpy())
    
    def act(self, obs):
        if np.random.uniform(0, 1) < self.e:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action
    
    def learn_batch(self):
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.rb.sample(self.minibatch_size)
        predict_q1 = self.pred_func(batch_obs)
        predict_q2 = predict_q1[torch.arange(len(predict_q1)), batch_action]
        target_q = batch_reward + (1 - batch_done) * self.gamma * self.target_func(batch_next_obs).max(1)[0]

        self.optimizer.zero_grad()
        l = self.loss(predict_q2, target_q)
        l.backward()
        self.optimizer.step()

    def syn_target(self):
        for target_param, param in zip(self.target_func.parameters(), 
                                       self.pred_func.parameters()):
            target_param.data.copy_(param.data)

    def learn(self, obs, action, reward, next_obs, done):
        self.global_steps += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) >= self.replay_start_size and self.global_steps % self.rb.num_steps == 0:
            self.learn_batch()
        if self.global_steps % self.target_update_interval == 0:
            self.syn_target()