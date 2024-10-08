import copy
import torch
import numpy as np

class Agent():
    def __init__(self, critic, critic_optimizer, actor, actor_optimizer, 
                 replay_buffer, replay_start_size, minibatch_size, 
                 target_update_interval, n_act, gamma, sigma, tau):
        self.global_steps = 0
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.critic_optimizer = critic_optimizer
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer

        self.rb = replay_buffer
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size

        self.target_update_interval = target_update_interval

        self.n_act = n_act
        self.gamma = gamma
        self.sigma = sigma #高斯噪声的标准差，均值为0
        self.tau = tau #目标网络的软更新参数

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        action = self.actor(obs).item() #因此此时action为一个数值，而不是离散动作索引，故不再需要随机采样
        #加上探索噪声
        action += self.sigma * np.random.randn(self.n_act)
        return action
    
    def learn_batch(self):
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.rb.sample(self.minibatch_size)

        #使用Actor目标网络获取下一时刻的action
        batch_next_action = self.target_actor(batch_next_obs)
        #使用Critic目标网络计算下一时刻状态选出的动作的q值
        next_q = self.target_critic(batch_next_obs, batch_next_action)
        #计算target_q值
        target_q = batch_reward + (1 - batch_done) * self.gamma * next_q
        #计算predict_q值
        predict_q = self.critic(batch_obs, batch_action)
        #计算损失并对critic训练网络的参数进行更新
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss(predict_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        #使用Actor训练网络获取当前时刻的action(不带噪声的)
        batch_action_ = self.actor(batch_obs)
        #使用Critic训练网络计算当前(s, a)的q值
        now_q = self.critic(batch_obs, batch_action_)
        #计算损失并对actor训练网络的参数进行更新
        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(now_q)
        actor_loss.backward()
        self.actor_optimizer.step()

    def syn_target(self):
        for target_param, param in zip(self.target_critic.parameters(), 
                                       self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), 
                                       self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, obs, action, reward, next_obs, done):
        self.global_steps += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) >= self.replay_start_size and self.global_steps % self.rb.num_steps == 0:
            self.learn_batch()
        if self.global_steps % self.target_update_interval == 0:
            self.syn_target()