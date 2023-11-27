import os
from unidiffusion.pipelines import UniDiffusionInferencePipeline
from unidiffusion.config import LazyConfig, default_argument_parser


def main(cfg, save_path):
    trainer = UniDiffusionInferencePipeline(cfg)
    trainer.save_diffusers(save_path)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    origin_config = LazyConfig.load('configs/common/get_common_config.py')
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.merge_additional_config(cfg, origin_config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    if cfg.train.resume is None:
        cfg.train.resume = 'latest'
    if args.config_file.endswith('yaml'):
        cfg.train.output_dir = os.path.dirname(args.config_file)
    cfg.only_inference = True
    main(cfg, args.save_diffusers_path)
