from pathlib import Path
import sys
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from huggingface_hub import login
login("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")
from unidiffusion.pipelines import SDXL_UniDiffusionTrainingPipeline
from unidiffusion.config import LazyConfig, default_argument_parser
def main(cfg):
    trainer = SDXL_UniDiffusionTrainingPipeline(cfg)
    trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    origin_config = LazyConfig.load('configs/common/get_common_config_sdxl.py')
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.merge_additional_config(cfg, origin_config) # 把origin_config的参数补到cfg中，cfg中已有的不变
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg)
