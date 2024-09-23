import torch

class Agent():
    def __init__(self, act_func, optimizer, gamma):
        self.act_func = act_func
        self.optimizer = optimizer
        
        self.gamma = gamma

    def act(self, obs): #因为采用了softmax，所以不需要epsilon-greedy
        probs = self.act_func(obs)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action
    
    def learn(self, obs_list, action_list, reward_list):
        G = 0 #记录该条episode的return
        self.optimizer.zero_grad() #在每一个episode前将梯度清零
        for i in reversed(range(len(obs_list))): #从后往前进行计算
            #获取每一步的reward
            reward = reward_list[i]
            #获取每一步的state
            obs = torch.FloatTensor(obs_list[i])
            #获取每一步的action
            action = action_list[i]
            #获取当前状态下每个action的概率
            probs = self.act_func(obs)
            #获取当前action概率的ln值
            log_prob = torch.log(probs[action])
            #计算当前时刻的q值 = reward + γ * 下一时刻的q值
            G = reward + self.gamma * G
            #计算每一步的损失函数
            loss = -log_prob * G
            #反向传播，梯度累加
            loss.backward()
        #梯度下降
        self.optimizer.step()