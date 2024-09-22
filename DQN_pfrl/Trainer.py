from Modules import MLP
import torch
import pfrl
import numpy as np

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
        explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=e, random_action_func=env.action_space.sample)
        rb = pfrl.replay_buffers.ReplayBuffer(capacity, num_steps)
        self.agent = pfrl.agents.DQN(
            q_function=q_func, 
            optimizer=optimizer, 
            explorer=explorer, 
            replay_buffer=rb, 
            minibatch_size=minibatch_size, 
            replay_start_size=replay_start_size, 
            target_update_interval=target_update_interval, 
            gamma=gamma, 
            phi=lambda x : x.astype(np.float32, copy=False))
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)

            self.agent.observe(obs, reward, done, done)

            obs = next_obs
            total_reward += reward

            if done: break

        return total_reward
    
    def test_episode(self, is_render=True):
        with self.agent.eval_mode():
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