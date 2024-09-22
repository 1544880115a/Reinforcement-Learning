from Trainer_simple import Trainer
import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    trainer = Trainer(env)
    trainer.train()