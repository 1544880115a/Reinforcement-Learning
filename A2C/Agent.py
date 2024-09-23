import torch

class Agent():
    def __init__(self, critic, actor, optimizer1, optimizer2, gamma):
        self.critic = critic
        self.optimizer1 = optimizer1
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.optimizer2 = optimizer2

        self.gamma = gamma

    def act(self, obs):
        probs = self.actor(obs)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action
    
    def learn(self, obs, action, reward, next_obs, done):
        predict_v = self.critic(obs)
        target_v = reward + (1 - float(done)) * self.gamma * self.critic(next_obs)
        td_error = target_v - predict_v

        probs = self.actor(obs)
        log_prob = torch.log(probs[action])

        critic_loss = self.loss(predict_v, target_v)
        actor_loss = -log_prob * td_error.detach()
        
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()

        critic_loss.backward()
        actor_loss.backward()

        self.optimizer1.step()
        self.optimizer2.step()