from Modules import Critic, Actor
from Agent import Agent
from ReplayBuffer import ReplayBuffer
import torch
import numpy as np

class Trainer():
    def __init__(self, env, episodes=300, critic_lr=0.01, actor_lr=0.001, 
                 alpha_lr=0.01, target_entropy=-1, target_update_interval=1, 
                 capacity=500, num_steps=1, minibatch_size=64, 
                 replay_start_size=200, gamma=0.98, tau=0.005):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n

        critic_1 = Critic(n_obs, n_act)
        critic1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=critic_lr)
        critic_2 = Critic(n_obs, n_act)
        critic2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=critic_lr)
        actor = Actor(n_obs, n_act)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        log_alpha.requires_grad = True
        log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)
        rb = ReplayBuffer(capacity, num_steps)

        self.agent = Agent(
            critic_1=critic_1, 
            critic1_optimizer=critic1_optimizer, 
            critic_2=critic_2, 
            critic2_optimizer=critic2_optimizer, 
            actor=actor, 
            actor_optimizer=actor_optimizer, 
            log_alpha=log_alpha, 
            log_alpha_optimizer=log_alpha_optimizer, 
            target_entropy=target_entropy, 
            target_update_interval=target_update_interval, 
            replay_buffer=rb, 
            minibatch_size=minibatch_size, 
            replay_start_size=replay_start_size, 
            gamma=gamma, 
            tau=tau)
        
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
    
    def test_episode(self, is_render):
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