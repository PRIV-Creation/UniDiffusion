
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from unidiffusion.config import LazyCall as L

train = {
    # common configs
    'project': 'UniDiffusion',
    'output_dir': './output',
    'pretrained_model_name_or_path': '',
    'revision': None,
    'seed': 0,
    'use_xformers': True,
    'gradient_checkpointing': False,
    'resume': None,  # "latest" | checkpoint path
    'use_ema': True,
    'resolution': 512,
    # pipelines configs
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
    'snr': {'enabled': False, 'snr_gamma': 5.0},
    # use dreambooth regularization
    'db': {
        'with_prior_preservation': False,
        "prior_generation_precision": 'fp16',
        'prior_loss_weight': 1.0,
        'class_data_dir': None,
        'class_prompt': None,
        'num_class_images': 100
    },
}

inference = {
    'save_path': None,
    'inference_iter': 5000,
    # 'batch_size': 1,    # not used
    'prompts': None,    # using dataset prompt if None
    'total_num': 10,
    'scheduler': 'DPMSolverMultistepScheduler',
    'pipeline_kwargs': {
        # arguments for pipeline.forward().
        'num_inference_steps': 25,
        'guidance_scale': 7.5,
    },

}

evaluation = {
    'evaluation_iter': 10000,
    'total_num': 1000,  # synthesis images num
    # 'batch_size': 1,    # not used
    'prompts': None,    # using dataset prompt if None
    'scheduler': 'DPMSolverMultistepScheduler',
    'pipeline_kwargs': {
        # arguments for pipeline.forward().
        'num_inference_steps': 25,
        'guidance_scale': 7.5,
    },
    'evaluator': {
        'fid': {'enabled': False, 'feature': 2048, 'real_image_num': 10000},
        'inception_score': {'enabled': False},
    },
}

accelerator = L(Accelerator)(
    gradient_accumulation_steps='${train.gradient_accumulation_iter}',
    mixed_precision='fp16',  # "no", "fp16", "bf16"
    project_config=L(ProjectConfiguration)(total_limit=None),
)
