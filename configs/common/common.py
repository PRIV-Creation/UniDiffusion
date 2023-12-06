from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from unidiffusion.config import LazyCall as L
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


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
    # Null-text Mode
    "null_text": False,
}

checkpoint = {
    'load_optimizer': True,
    'load_scheduler': True,
}

inference_pipeline = L(StableDiffusionPipeline.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
)

inference = {
    'save_path': None,
    'skip_error': False,
    'inference_iter': 5000,
    'rectify_uncond': False,
    'prompts': None,    # string or prompt file path. Using randomly selected dataset prompts if None.
    'total_num': 10,
    'scheduler': L(DPMSolverMultistepScheduler.from_config)(),
    'forward_kwargs': {
        # arguments for pipeline.forward().
        'num_inference_steps': 25,
        'guidance_scale': 7.5,
    },

}

evaluation = {
    'evaluation_iter': 10000,
    'skip_error': False,
    'total_num': 1000,  # synthesis images num
    'rectify_uncond': False,
    # 'batch_size': 1,    # not used
    'prompts': None,    # using dataset prompt if None
    'scheduler': L(DPMSolverMultistepScheduler.from_config)(),
    'forward_kwargs': {
        # arguments for pipeline.forward().
        'num_inference_steps': 25,
        'guidance_scale': 7.5,
    },
    "save_image": False,
    "save_path": None,
    'evaluator': {
        'fid': {'enabled': False, 'feature': 2048, 'real_image_num': 10000},
        'inception_score': {'enabled': False},
        'clip_score': {
            'enabled': False,
            'clip_model': 'openai/clip-vit-large-patch14',
            'prompts': None,      # used for generation. use evaluation.prompts if None.
            'prompts_ori': None,  # used for calculate text-image similarity. use evaluation.evaluator.clip_score.prompts if None.
            'total_num': None,     # only used when prompts is not None.
        },
    },
}

accelerator = L(Accelerator)(
    gradient_accumulation_steps='${train.gradient_accumulation_iter}',
    mixed_precision='fp16',  # "no", "fp16", "bf16"
    project_config=L(ProjectConfiguration)(total_limit=None),
)
