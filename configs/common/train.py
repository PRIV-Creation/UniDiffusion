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
from diffusion_trainer.config import LazyCall as L

train = {
    # common configs
    'output_dir': './output',
    'pretrained_model_name_or_path': '',
    'revision': '',
    'seed': 0,
    'use_xformers': True,
    # training configs
    'max_iter': 10000,
    'max_grad_norm': 1.0,
    'lr_warmup_iter': 0,
    'gradient_accumulation_iter': 1,
    # logging configs
    'checkpointing_iter': 5000,
    # wandb
    'wandb': {
        'enabled': False,
        'project': 'DiffusionTrainer',
        'entity': None,
    },
}

accelerator = L(Accelerator)(
    gradient_accumulation_steps=1,
    mixed_precision='fp16',  # "no", "fp16", "bf16"
    log_with='wandb',  # Supported platforms are tensorboard, wandb and comet_ml. Use all to report to all integrations.
    # logging_dir=os.path.join(train['output_dir'], 'logs'),
    project_config=L(ProjectConfiguration)(total_limit=None),
)
