from Trainer import Trainer
import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    trainer = Trainer(env)
    trainer.train()