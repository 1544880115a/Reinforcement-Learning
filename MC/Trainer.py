from Agent import Agent

class Trainer():
    def __init__(self, env, episodes=1000, episode_max_len=1000, gamma=0.9, epsilon=0.1):
        self.env = env
        self.episodes = episodes
        self.episode_max_len = episode_max_len

        n_obs = env.observation_space.n
        n_act = env.action_space.n

        self.agent = Agent(
            n_obs=n_obs, 
            n_act=n_act, 
            gamma=gamma, 
            epsilon=epsilon)
    
    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs_list, action_list, reward_list = [], [], []
        
        for i in range(self.episode_max_len):
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)

            obs = next_obs
            total_reward += reward

            if done: break
        
        self.agent.learn(obs_list, action_list, reward_list)
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
        print(f'test_reward={test_reward:F}')