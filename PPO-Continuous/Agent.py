import torch

class Agent():
    def __init__(self, critic, critic_optimizer, actor, 
                 actor_optimizer, gamma, lmbda, eps, epochs):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.actor_optimizer = actor_optimizer
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs

    def act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        #输出当前动作概率的高斯分布
        u, std = self.actor(obs)
        #构造高斯分布
        action_dict = torch.distributions.Normal(u, std)
        #选择动作
        action = action_dict.sample().item()
        return [action]
    
    def learn(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        obs_list = torch.FloatTensor(obs_list)
        action_list = torch.tensor(action_list).view(-1, 1)
        reward_list = torch.FloatTensor(reward_list).view(-1, 1)
        next_obs_list = torch.FloatTensor(next_obs_list)
        done_list = torch.FloatTensor(done_list).view(-1, 1)

        #计算下一时刻状态的v值
        next_v = self.critic(next_obs_list)
        #计算Critic网络的target
        target_v = reward_list + (1 - done_list) * self.gamma * next_v
        #计算当前时刻状态的v值
        predict_v = self.critic(obs_list)
        #计算TD error
        td_error = target_v - predict_v

        #计算GAE
        td_error = td_error.detach().numpy()
        advantage_list = []
        advantage = 0
        for err in td_error[::-1]:
            advantage = self.lmbda * self.gamma * advantage + err
            advantage_list.append(advantage)
        #正序
        advantage_list.reverse()
        advantage = torch.FloatTensor(advantage_list).view(-1, 1)

        #使用Actor网络计算出当前状态选择动作的高斯分布
        u, std = self.actor(obs_list)
        #基于均值和标准差构造正态分布
        action_dict = torch.distributions.Normal(u.detach(), std.detach())
        #从正态分布中选择动作，并使用log函数
        old_log_prob = action_dict.log_prob(action_list)

        #一个episode的数据训练epochsci
        for _ in range(self.epochs):
            #使用新参数预测当前状态下的动作
            u, std = self.actor(obs_list)
            #构造正态分布
            action_dict = torch.distributions.Normal(u, std)
            #采取行为的概率
            log_prob = action_dict.log_prob(action_list)
            #计算新旧概率的比值
            ratio = torch.exp(log_prob - old_log_prob)

            #公式的左侧项
            surr1 = ratio * advantage
            #公式的右侧项
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            #Actor网络的损失
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            #Critic网络的损失
            critic_loss = self.loss(self.critic(obs_list), target_v.detach())

            #优化参数
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            critic_loss.backward()
            actor_loss.backward()

            self.critic_optimizer.step()
            self.actor_optimizer.step()