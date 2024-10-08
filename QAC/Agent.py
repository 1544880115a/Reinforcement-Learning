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
    
    def learn(self, obs, action, reward, next_obs, next_action, done):
        predict_q1 = self.critic(obs)[action]
        target_q = reward + (1 - float(done)) * self.gamma * self.critic(next_obs)[next_action]

        self.critic_optimizer.zero_grad()
        critic_loss = self.loss(predict_q1, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        probs = self.actor(obs)
        log_prob = torch.log(probs[action])
        predict_q2 = self.critic(obs)[action]

        self.actor_optimizer.zero_grad()
        actor_loss = -log_prob * predict_q2.detach()
        actor_loss.backward()
        self.actor_optimizer.step()