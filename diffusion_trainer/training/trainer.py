import diffusers
import os
from diffusion_trainer.config import LazyConfig, instantiate, default_argument_parser
import itertools
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from diffusers.training_utils import EMAModel
from diffusion_trainer.utils.checkpoint import save_model_hook, load_model_hook


logger = get_logger(__name__)


class DiffusionTrainer:
    tokenizer = None
    noise_scheduler = None
    text_encoder = None
    vae = None
    unet = None
    unet_ema = None
    weight_dtype = None

    def __init__(self, cfg, training):
        self.accelerator = None
        self.optimizer = None
        self.models = None
        self.cfg = cfg
        self.training = training

        self.default_setup()
        self.build_model()
        if training:
            self.build_dataset()
            self.build_optimizer()
            self.build_scheduler()
            self.build_evaluator()
            self.build_criterion()
            self.prepare_training()

    def default_setup(self):
        self.accelerator = instantiate(self.cfg.accelerator)
        logger.info(self.accelerator.state, main_process_only=True)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.train.seed is not None:
            set_seed(self.cfg.train.seed)

        if self.accelerator.is_main_process:
            os.makedirs(self.cfg.train.output_dir, exist_ok=True)

        # mixed precision
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def build_dataset(self):
        pass

    def build_model(self):
        # update config if load checkpoint
        # ...

        # build models and noise scheduler
        self.vae = instantiate(self.cfg.vae)
        self.tokenizer = instantiate(self.cfg.tokenizer)
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler)
        self.text_encoder = instantiate(self.cfg.text_encoder)
        self.unet = instantiate(self.cfg.unet)
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler)

        self.models = [self.vae, self.tokenizer, self.text_encoder, self.unet]

    def build_optimizer(self):
        # get trainable parameters
        trainable_params = []
        for model in self.models:
            p = model.get_trainable_params()
            if p is not None:
                trainable_params.append(p)
        trainable_params = itertools.chain(*trainable_params)

        # build optimizer
        self.cfg.optimizer.params = trainable_params
        self.optimizer = instantiate(self.cfg.optimizer)

        # print num of trainable parameters
        num_params = sum([p.numel() for params_group in self.optimizer.param_groups for p in params_group ['params']])
        logger.info(f"Number of trainable parameters: {num_params / 1e6} M", main_process_only=True)

    def build_scheduler(self):
        pass

    def build_evaluator(self):
        pass

    def build_criterion(self):
        pass

    def prepare_training(self):
        # prepare models
        for model in self.models:
            model = self.accelerator.prepare(model)
        if not self.vae.trainable:
            self.vae.requires_grad_(False)
            self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if not self.unet.trainable:
            self.unet.requires_grad_(False)
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        if not self.text_encoder.trainable:
            self.unet.requires_grad_(False)

        # prepare xformers
        if self.cfg.train.use_xformers and self.unet.trainable:
            self.unet.enable_xformers_memory_efficient_attention()

        # prepare tracker
        if self.accelerator.is_main_process:
            wandb_kwargs = {
                'entity': self.cfg.wandb_entity,
                'name': os.path.split(self.cfg.output_dir)[-1] if os.path.split(self.cfg.output_dir)[-1] != '' else
                os.path.split(self.cfg.output_dir)[-2],
            }
            self.accelerator.init_trackers(self.cfg.wandb.project, config=vars(self.cfg),
                                           init_kwargs={'wandb': wandb_kwargs})

        # prepare checkpoint hook
        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def train(self):
        pass

    def inference(self):
        pass

    def evaluate(self):
        pass

