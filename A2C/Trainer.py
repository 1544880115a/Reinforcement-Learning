from Modules import Critic, Actor
from Agent import Agent
import torch

class Trainer():
    def __init__(self, env, episodes=700, lr1=0.01, lr2=0.001, gamma=0.9):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n

        critic = Critic(n_obs)
        optimizer1 = torch.optim.AdamW(critic.parameters(), lr=lr1)
        actor = Actor(n_obs, n_act)
        optimizer2 = torch.optim.AdamW(actor.parameters(), lr=lr2)
        

        self.agent = Agent(
            critic=critic, 
            actor=actor, 
            optimizer1=optimizer1, 
            optimizer2=optimizer2, 
            gamma=gamma)
    
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
    
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)

            self.agent.learn(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            if done: break

        return total_reward
    
    def test_episode(self, is_render=True):
        total_reward = 0
        obs = self.env.reset()

        while True:
            obs = torch.FloatTensor(obs)
            action = self.agent.act(obs)

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