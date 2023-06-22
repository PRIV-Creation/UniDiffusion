
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
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
    'resolution': 512,
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
    'batch_size': 1,    # not used
    'prompts': None,    # using dataset prompt if None
    'total_num': 10,
    'scheduler': 'DPMSolverMultistepScheduler',
    'num_inference_steps': 25,
    'guidance_scale': 1.0,
}

evaluation = {
    'evaluation_iter': 10000,
    'total_num': 1000,  # synthesis images num
    'batch_size': 4,
    'prompts': None,    # using dataset prompt if None
    'scheduler': 'DPMSolverMultistepScheduler',
    'num_inference_steps': 25,
    'guidance_scale': 1.0,
    'evaluator': {
        'fid': {'enabled': False, 'feature': 2048, 'real_image_num': 10000},
        'inception_score': {'enabled': False},
    }
}

accelerator = L(Accelerator)(
    gradient_accumulation_steps='${train.gradient_accumulation_iter}',
    mixed_precision='fp16',  # "no", "fp16", "bf16"
    project_config=L(ProjectConfiguration)(total_limit=None),
)
