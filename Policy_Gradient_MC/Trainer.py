from Modules import MLP
from Agent import Agent
import torch

class Trainer():
    def __init__(self, env, episodes=1000, lr=0.001, gamma=0.9):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n

        act_func = MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(act_func.parameters(), lr=lr)

        self.agent = Agent(
            act_func=act_func, 
            optimizer=optimizer, 
            gamma=gamma)
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        obs_list, action_list, reward_list = [], [], []

        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)

            obs = next_obs
            total_reward += reward
            
            if done: break
        
        self.agent.learn(obs_list, action_list, reward_list)
        return total_reward
    
    def test_episode(self, is_render=True):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)

        while True:
            action = self.agent.act(obs)

            obs, reward, done, _ = self.env.step(action)
            obs = torch.FloatTensor(obs)

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