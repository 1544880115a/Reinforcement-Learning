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
    
    def learn(self, obs, action, reward, next_obs, next_action, done):
        predict_q1 = self.critic(obs)[action]
        target_q = reward + (1 - float(done)) * self.gamma * self.critic(next_obs)[next_action]

        self.optimizer1.zero_grad()
        critic_loss = self.loss(predict_q1, target_q)
        critic_loss.backward()
        self.optimizer1.step()

        probs = self.actor(obs)
        log_prob = torch.log(probs[action])
        predict_q2 = self.critic(obs)[action]

        self.optimizer2.zero_grad()
        actor_loss = -log_prob * predict_q2.detach()
        actor_loss.backward()
        self.optimizer2.step()