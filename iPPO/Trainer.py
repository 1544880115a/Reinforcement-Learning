from Modules import Critic, Actor
from Agent import Agent
import torch
import time

class Trainer():
    def __init__(self, env, episodes=500, critic_lr=0.001, actor_lr=0.0003, 
                 gamma=0.9, lmbda=0.9, eps=0.2):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space[0].shape[0]
        n_act = env.action_space[0].n

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
            eps=eps)
        
    def train_episode(self):
        obs1, obs2 = self.env.reset()
        obs_list1, action_list1, reward_list1, next_obs_list1, done_list1 = [], [], [], [], []
        obs_list2, action_list2, reward_list2, next_obs_list2, done_list2 = [], [], [], [], []

        while True:
            action1, action2 = self.agent.act(obs1), self.agent.act(obs2)

            next_obs, reward, done, _ = self.env.step([action1, action2])

            obs_list1.append(obs1)
            action_list1.append(action1)
            reward_list1.append(reward[0])
            next_obs_list1.append(next_obs[0])
            done_list1.append(False)
            
            obs_list2.append(obs2)
            action_list2.append(action2)
            reward_list2.append(reward[1])
            next_obs_list2.append(next_obs[1])
            done_list2.append(False)

            obs1, obs2 = next_obs
            
            if all(done): break

        self.agent.learn(obs_list1, action_list1, reward_list1, next_obs_list1, done_list1)
        self.agent.learn(obs_list2, action_list2, reward_list2, next_obs_list2, done_list2)

    def test_episode(self):
        obs1, obs2 = self.env.reset()

        while True:
            action1, action2 = self.agent.act(obs1), self.agent.act(obs2)

            next_obs, _, done, _ = self.env.step([action1, action2])

            obs1, obs2 = next_obs
            self.env.render()
            time.sleep(1)

            if all(done): break

    def train(self):
        for e in range(self.episodes):
            self.train_episode()
            print(f'train in episode {e}')
        self.test_episode()