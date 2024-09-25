from Trainer import Trainer
from ma_gym.envs.combat.combat import Combat

if __name__ == '__main__':
    env = Combat(grid_shape=(15, 15), n_agents=2, n_opponents=2)
    trainer = Trainer(env)
    trainer.train()