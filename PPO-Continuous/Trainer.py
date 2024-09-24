from Modules import Critic, Actor
from Agent import Agent
import torch

class Trainer():
    def __init__(self, env, episodes=1500, critic_lr=0.01, actor_lr=0.001, 
                 gamma=0.98, lmbda=0.95, eps=0.2, epochs=10):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]

        critic = Critic(n_obs)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        actor = Actor(n_obs, n_act)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        self.agent = Agent(
            critic=critic, 
            critic_optimizer=critic_optimizer, 
            actor=actor, 
            actor_optimizer=actor_optimizer, 
            gamma=gamma, 
            lmbda=lmbda, 
            eps=eps, 
            epochs=epochs)
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []

        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)

            obs = next_obs
            total_reward += reward

            if done: break
        
        return total_reward
    
    def test_episode(self, is_render=False):
        total_reward = 0
        obs = self.env.reset()

        while True:
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