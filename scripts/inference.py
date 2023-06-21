from unidiffusion.training.trainer import DiffusionTrainer
from unidiffusion.config import LazyConfig, default_argument_parser


def main(cfg):
    trainer = DiffusionTrainer(cfg, training=True)
    trainer.inference()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg)

