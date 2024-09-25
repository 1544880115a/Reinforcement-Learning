import torch

class Agent():
    def __init__(self, critic, critic_optimizer, actor, 
                 actor_optimizer, gamma, lmbda, eps):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.actor_optimizer = actor_optimizer

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps

    def act(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.actor(obs)
        action_dict = torch.distributions.Categorical(probs)
        action = action_dict.sample().item()
        return action
    
    def learn(self, obs_list, action_list, reward_list, next_obs_list, done_list):
        obs_list = torch.FloatTensor(obs_list)
        action_list = torch.IntTensor(action_list)
        reward_list = torch.FloatTensor(reward_list).view(-1, 1)
        next_obs_list = torch.FloatTensor(next_obs_list)
        done_list = torch.FloatTensor(done_list).view(-1, 1)

        #计算下一时刻的v值
        next_v = self.critic(next_obs_list)
        #critic网络的target
        target_v = reward_list + (1 - done_list) * self.gamma * next_v
        #当前时刻的v值
        predict_v = self.critic(obs_list)
        #TD error
        td_error = target_v - predict_v

        #计算优势函数GAE
        td_error = td_error.detach().numpy()
        advantage = 0
        advantage_list = []
        for err in td_error[::-1]: #逆序计算
            advantage = self.lmbda * self.gamma * advantage + err
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.FloatTensor(advantage_list)

        #计算当前状态动作概率
        old_probs = self.actor(obs_list)
        log_old_probs = torch.log(old_probs[torch.arange(len(old_probs)), action_list]).detach()

        probs = self.actor(obs_list)
        log_probs = torch.log(probs[torch.arange(len(probs)), action_list])
        ratio = log_probs / log_old_probs
        #公式的左侧
        surr1 = ratio * advantage
        #公式的右侧
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

        #计算损失
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = self.loss(self.critic(obs_list), target_v.detach())

        #参数更新
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward()
        actor_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()