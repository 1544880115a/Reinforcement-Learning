import torch

class Agent():
    def __init__(self, critic, critic_optimizer, 
                 actor, actor_optimizer, gamma, lmbda, eps, epochs):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.gamma = gamma
        self.lmbda = lmbda #GAE中的λ
        self.eps = eps #截断超参数
        self.epochs = epochs #每个episode的数据训练的轮数

    def act(self, obs):
        probs = self.actor(obs)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action
    
    def learn(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        obs_list = torch.FloatTensor(obs_list)
        action_list = torch.IntTensor(action_list)
        reward_list = torch.FloatTensor(reward_list).view(-1, 1)
        next_obs_list = torch.FloatTensor(next_obs_list)
        done_list = torch.FloatTensor(done_list).view(-1, 1)
        #计算下一个时刻状态的v值
        next_v = self.critic(next_obs_list)
        #critic网络的target
        target_v = reward_list + (1 - done_list) * self.gamma * next_v
        #预测的当前状态的v值
        predict_v = self.critic(obs_list)
        #目标v值和预测v值之间的state value之差
        td_error = target_v - predict_v
        #优势函数初始化
        advantage = 0
        advantage_list = []

        #计算优势函数
        td_error = td_error.detach().numpy()
        for err in td_error[::-1]: #逆序时序差分值，即从T=t-1往前计算
            #优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + err
            advantage_list.append(advantage)
        #将优势函数列表正序
        advantage_list.reverse()
        #将其转换为tensor形式
        advantage = torch.FloatTensor(advantage_list)

        #actor网络给出每个old动作的概率
        old_probs = self.actor(obs_list)
        old_log_probs = torch.log(old_probs[torch.arange(len(old_probs)), action_list]).detach()

        #一个episode得到的数据训练epochs轮
        for _ in range(self.epochs):
            #每一轮更新一次Actor网络
            probs = self.actor(obs_list)
            log_probs = torch.log(probs[torch.arange(len(probs)), action_list])
            #计算新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            #近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            #公式的右端项
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            #Actor网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            #Critic网络的损失函数
            critic_loss = self.loss(self.critic(obs_list), target_v.detach())

            #进行参数更新
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()

            critic_loss.backward()
            actor_loss.backward()

            self.critic_optimizer.step()
            self.actor_optimizer.step()