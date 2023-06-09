from diffusion_trainer.trainer import DiffusionTrainer
from diffusion_trainer.configs import Config
from diffusion_trainer.config import LazyConfig, instantiate

if __name__ == '__main__':
    cfg = Config()
    trainer = DiffusionTrainer(cfg, training=True)
    trainer.train()