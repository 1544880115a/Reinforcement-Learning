from ReplayBuffer import ReplayBuffer
from Agent import Agent
from Modules import Critic, Actor
import torch

class Trainer():
    def __init__(self, env, episodes=20, critic_lr=0.003, actor_lr=0.0003, 
                 capacity=1000, num_steps=1, replay_start_size=200, 
                 minibatch_size=64, target_update_interval=1, 
                 gamma=0.98, sigma=0.01, tau=0.005):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]
        action_bound = env.action_space.high[0]

        critic = Critic(n_obs, n_act)
        critic_optimizer = torch.optim.Adam(critic.parameters(), critic_lr)
        actor = Actor(n_obs, n_act, action_bound)
        actor_optimizer = torch.optim.Adam(actor.parameters(), actor_lr)
        rb = ReplayBuffer(capacity, num_steps)

        self.agent = Agent(
            critic=critic, 
            critic_optimizer=critic_optimizer, 
            actor=actor,  
            actor_optimizer=actor_optimizer, 
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

            self.agent.learn(obs, action, reward, next_obs, done)

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