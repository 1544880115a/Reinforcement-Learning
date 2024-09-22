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
    
    def train(self):
        pfrl.experiments.train_agent_with_evaluation(
            agent=self.agent, 
            env=self.env, 
            steps=20000, #总共训练20000步
            eval_n_steps=None, #eval_n_steps与eval_n_episodes二选一，一个是每次评估多少步，一个是每次评估多少episode
            eval_n_episodes=10, #每次评估10个episode
            train_max_episode_len=200, #每次训练最大的步数
            eval_interval=1000, #每训练1000步进行一次评估
            outdir='./DQN_pfrl/result')