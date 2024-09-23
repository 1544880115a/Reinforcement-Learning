from ReplayBuffer import ReplayBuffer
from Agent import Agent
from Modules import Critic, Actor
import torch
import gym

class Trainer():
    def __init__(self, env, episodes=1000, lr1=0.003, lr2=0.0003, 
                 capacity=1000, num_steps=1, replay_start_size=200, 
                 minibatch_size=64, target_update_interval=1, 
                 gamma=0.98, sigma=0.01, tau=0.005):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        critic = Critic(n_obs, n_act)
        optimizer1 = torch.optim.Adam(critic.parameters(), lr=lr1)
        actor = Actor(n_obs, n_act, action_bound)
        optimizer2 = torch.optim.Adam(actor.parameters(), lr=lr2)
        rb = ReplayBuffer(capacity, num_steps)

        self.agent = Agent(
            critic=critic, 
            optimizer1=optimizer1, 
            actor=actor,  
            optimizer2=optimizer2, 
            replay_buffer=rb, 
            replay_start_size=replay_start_size, 
            minibatch_size=minibatch_size, 
            target_update_interval=target_update_interval, 
            n_act=n_act, 
            gamma=gamma, 
            sigma=sigma, 
            tau=tau)
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)

            