from Agent import Agent

class Trainer():
    def __init__(self, env, episodes=1000, lr=0.1, gamma=0.9, e=0.1):
        self.env = env
        self.episodes = episodes

        n_obs = env.observation_space.n
        n_act = env.action_space.n

        self.agent = Agent(
            n_obs=n_obs, 
            n_act=n_act, 
            lr=lr, 
            gamma=gamma, 
            e=e)
        
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        action = self.agent.act(obs)

        while True:
            next_obs, reward, done, _ = self.env.step(action)
            next_action = self.agent.act(next_obs)

            self.agent.learn(obs, action, reward, next_obs, next_action, done)

            obs = next_obs
            action = next_action
            total_reward += reward

            if done: break
        
        return total_reward
    
    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()

        while True:
            action = self.agent.predict(obs)

            obs, reward, done, _ = self.env.step(action)

            total_reward += reward
            self.env.render()

            if done: break
        
        return total_reward
    
    def train(self):
        for e in range(self.episodes):
            ep_reward = self.train_episode()
            print(f'episode {e}, reward={ep_reward:f}')
        test_reward = self.test_episode()
        print(f'test_reward={test_reward:f}')