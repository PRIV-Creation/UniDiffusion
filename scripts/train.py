from diffusion_trainer.training.trainer import DiffusionTrainer
from diffusion_trainer.config import LazyConfig, default_argument_parser
from diffusion_trainer.utils.logger import setup_logger


def main(cfg):
    trainer = DiffusionTrainer(cfg, training=True)
    trainer.train()


if __name__ == '__main__':
    setup_logger()
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg)

