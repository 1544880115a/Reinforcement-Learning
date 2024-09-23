from Trainer import Trainer
import gym

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    trainer = Trainer(env)
    trainer.train()