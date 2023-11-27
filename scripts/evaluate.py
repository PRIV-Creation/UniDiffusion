import os
from unidiffusion.pipelines import UniDiffusionEvaluationPipeline
from unidiffusion.config import LazyConfig, default_argument_parser


def main(cfg):
    trainer = UniDiffusionEvaluationPipeline(cfg)
    trainer.evaluate()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    if cfg.train.resume is None:
        cfg.train.resume = 'latest'
    if args.config_file.endswith('yaml'):
        cfg.train.output_dir = os.path.dirname(args.config_file)
    cfg.only_inference = True
    main(cfg)

