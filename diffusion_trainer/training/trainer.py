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


logger = get_logger(__name__)


class DiffusionTrainer:
    tokenizer = None
    noise_scheduler = None
    text_encoder = None
    vae = None
    unet = None
    unet_ema = None

    def __init__(self, cfg, training):
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

    def default_setup(self):
        accelerator = instantiate(self.cfg.accelerator)
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.train.seed is not None:
            set_seed(self.cfg.train.seed)

        if accelerator.is_main_process:
            os.makedirs(self.cfg.train.output_dir, exist_ok=True)

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

    def train(self):
        pass

    def inference(self):
        pass

    def evaluate(self):
        pass

