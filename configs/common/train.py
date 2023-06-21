import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from diffusers.training_utils import EMAModel
from unidiffusion.config import LazyCall as L

train = {
    # common configs
    'project': 'DiffusionTrainer',
    'output_dir': './output',
    'pretrained_model_name_or_path': '',
    'revision': None,
    'seed': 0,
    'use_xformers': True,
    'gradient_checkpointing': False,
    'resume': None,  # "latest" | checkpoint path
    'use_ema': True,
    # training configs
    'max_iter': 10000,
    'max_grad_norm': 1.0,
    'lr_warmup_iter': 0,
    'gradient_accumulation_iter': 1,
    # logging configs
    'checkpointing_iter': 5000,
    # Experiment Trackers
    'wandb': {'enabled': False, 'entity': None,},
    'tensorboard': {'enabled': False},
    'comet_ml': {'enabled': False},
    # training mechanisms
    'snr': {'enabled': False, 'snr_gamma': 5.0}
}

inference = {
    'inference_iter': 5000,
    'batch_size': 1,
    'prompts': None,    # using dataset prompt if None
    'total_num': 10,
}

accelerator = L(Accelerator)(
    gradient_accumulation_steps=1,
    mixed_precision='fp16',  # "no", "fp16", "bf16"
    # logging_dir=os.path.join(train['output_dir'], 'logs'),
    project_config=L(ProjectConfiguration)(total_limit=None),
)
