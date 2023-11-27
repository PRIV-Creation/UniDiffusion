from unidiffusion.pipelines import UniDiffusionTrainingPipeline
from unidiffusion.config import LazyConfig, default_argument_parser


def main(cfg):
    trainer = UniDiffusionTrainingPipeline(cfg)
    trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    origin_config = LazyConfig.load('configs/common/get_common_config.py')
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.merge_additional_config(cfg, origin_config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg)
