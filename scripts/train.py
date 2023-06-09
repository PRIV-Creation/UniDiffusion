from diffusion_trainer.trainer import DiffusionTrainer
from diffusion_trainer.config import LazyConfig, instantiate, default_argument_parser


def main(cfg):
    trainer = DiffusionTrainer(cfg, training=True)
    trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg)

