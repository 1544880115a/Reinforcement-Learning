import torch
import copy

class Agent():
    def __init__(self, critic_1, critic1_optimizer, critic_2, 
                 critic2_optimizer, actor, actor_optimizer, 
                 log_alpha, log_alpha_optimizer, target_entropy, 
                 target_update_interval, replay_buffer, minibatch_size, 
                 replay_start_size, gamma, tau):
        self.global_steps = 0
        #策略网络
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        #价值网络
        self.critic_1 = critic_1
        self.critic1_optimizer = critic1_optimizer
        self.critic_2 = critic_2
        self.critic2_optimizer = critic2_optimizer
        self.loss = torch.nn.MSELoss()

        #目标网络
        self.target_critic_1 = copy.deepcopy(critic_1)
        self.target_critic_2 = copy.deepcopy(critic_2)

        #熵正则化系数α的对数值
        self.log_alpha = log_alpha
        #熵正则化系数α对数值的优化器
        self.log_alpha_optimizer = log_alpha_optimizer
        #想要达到的熵值，用于对α进行更新
        self.target_entropy = target_entropy

        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.tau = tau #软更新目标网络，与DDPG一样

        self.rb = replay_buffer
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size

    def act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.actor(obs)
        action_dict = torch.distributions.Categorical(probs)
        action = action_dict.sample().item()
        return action
    
    def syn_target(self):
        for target_param, param in zip(self.target_critic_1.parameters(), 
                                       self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), 
                                       self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def learn_batch(self):
        # --------------------------#
        # 更新两个价值网络
        # --------------------------#
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.rb.sample(self.minibatch_size)
        #Actor网络输出下一时刻状态下的动作概率 [batch_size, n_act]
        next_probs = self.actor(batch_next_obs)
        #对每个动作的概率计算ln, 加入一个值是防止出现概率为0的action [batch_size, n_act]
        log_next_probs = torch.log(next_probs + 1e-8)
        #计算熵 [batch_size, 1]
        entropy = -torch.sum(next_probs * log_next_probs, dim=1, keepdims=True)
        #使用两个目标价值网络，计算下一时刻的Q值 [batch_size, n_act]
        next_q1_value = self.target_critic_1(batch_next_obs)
        next_q2_value = self.target_critic_2(batch_next_obs)
        #取出最小的q值，计算下个时刻的v值，v值等于所有q值的期望，所以乘概率求和，[batch_size, 1]
        min_qvalue = torch.sum(next_probs * torch.min(next_q1_value, next_q2_value), dim=1, keepdim=True)
        #下一个时刻的v值+熵值 [batch_size, 1]
        next_v = min_qvalue + self.log_alpha.exp() * entropy
        #计算出目标v值
        target_v = batch_reward + (1 - batch_done) * self.gamma * next_v
        #对两个软Q网络进行更新，分别用两个软Q网络计算当前时刻的Q值，与刚刚计算出的目标V值计算损失
        predict_q = self.critic_1(batch_obs)
        predict_q1 = predict_q[torch.arange(len(predict_q)), batch_action]
        predict_q = self.critic_2(batch_obs)
        predict_q2 = predict_q[torch.arange(len(predict_q)), batch_action]
        #计算损失
        critic_1_loss = self.loss(predict_q1, target_v.detach())
        critic_2_loss = self.loss(predict_q2, target_v.detach())
        #参数更新
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # --------------------------#
        # 更新策略网络
        # --------------------------#
        #Actor网络输出当前时刻状态的动作概率 [batch_size, action]
        probs = self.actor(batch_obs)
        #对每个动作的概率计算ln [batch_size, action]
        log_probs = torch.log(probs + 1e-8)
        #计算熵 [b, 1]
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        #使用两个价值网络获取当前时刻的Q值
        q1_value = self.critic_1(batch_obs)
        q2_value = self.critic_2(batch_obs)
        #取出最小的q值，计算当前时刻的v值，v值等于所有q值的期望，所以乘概率求和，[batch_size, 1]
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        #计算策略网络的损失
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        #参数更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------#
        # 更新熵正则化系数α
        # --------------------------#
        #计算损失
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        #参数更新
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_steps += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if len(self.rb) >= self.replay_start_size and self.global_steps % self.rb.num_steps == 0:
            self.learn_batch()
        if self.global_steps % self.target_update_interval == 0:
            self.syn_target()