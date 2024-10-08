import torch

class Agent():
    def __init__(self, critic, actor, critic_optimizer, actor_optimizer, gamma):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.loss = torch.nn.MSELoss()

        self.actor = actor
        self.actor_optimizer = actor_optimizer

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
        
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward()
        actor_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()