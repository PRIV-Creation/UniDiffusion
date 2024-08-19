from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def main(cfg):
    trainer = UniDiffusionInferencePipeline(cfg)
    trainer.inference()

import os
from unidiffusion.pipelines import UniDiffusionInferencePipeline
from unidiffusion.config import LazyConfig, default_argument_parser



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

