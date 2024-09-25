import collections
import random
import torch

class ReplayBuffer():
    def __init__(self, capacity, num_steps):
        self.buffer = collections.deque(maxlen=capacity)
        self.num_steps = num_steps

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, minibatch_size):
        mini_batch = random.sample(self.buffer, minibatch_size)
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = zip(*mini_batch)
        batch_obs = torch.FloatTensor(batch_obs)
        batch_action = torch.IntTensor(batch_action)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1)
        batch_next_obs = torch.FloatTensor(batch_next_obs)
        batch_done = torch.FloatTensor(batch_done).view(-1, 1)
        return batch_obs, batch_action, batch_reward, batch_next_obs, batch_done
    
    def __len__(self):
        return len(self.buffer)