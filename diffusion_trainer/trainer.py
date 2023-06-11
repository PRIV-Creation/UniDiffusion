import logging
import diffusers
import os
from accelerate.logging import get_logger
from diffusion_trainer.config import LazyConfig, instantiate, default_argument_parser

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
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
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
        # tokenizer
        self.vae = instantiate(self.cfg.vae)
        self.tokenizer = instantiate(self.cfg.tokenizer)
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler)
        self.text_encoder = instantiate(self.cfg.text_encoder)

        self.unet = instantiate(self.cfg.unet)

    def build_optimizer(self):
        pass

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

