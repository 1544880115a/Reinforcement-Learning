from Modules import MLP
from Agent import Agent
from ReplayBuffer import ReplayBuffer
import torch

class Trainer():
    def __init__(self, env, episodes=1000, capacity=2000, 
                 num_steps=4, minibatch_size=32, replay_start_size=200, 
                 target_update_interval=200, lr=0.001, gamma=0.9, e=0.1):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n

        q_func = MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = ReplayBuffer(capacity, num_steps)
        self.agent = Agent(
            q_func=q_func, 
            optimizer=optimizer, 
            replay_buffer=rb, 
            minibatch_size=minibatch_size, 
            replay_start_size=replay_start_size, 
            target_update_interval=target_update_interval, 
            n_act=n_act, 
            gamma=gamma, 
            e=e)
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)

            self.agent.learn(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            if done: break

        return total_reward
    
    def test_episode(self, is_render=True):
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.predict(obs)

            obs, reward, done, _ = self.env.step(action)

            total_reward += reward
            if is_render: self.env.render()

            if done: break
        
        return total_reward
    
    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print(f'episode {e}, reward={ep_reward:f}')
        
        test_reward = 0
        is_render = False
        for i in range(5):
            if i == 4: is_render = True
            test_reward += self.test_episode(is_render)
        print(f'test_reward={test_reward / 5:f}')