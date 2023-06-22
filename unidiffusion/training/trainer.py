import diffusers
import os
import numpy as np
import wandb
from unidiffusion.config import instantiate
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from diffusers.training_utils import EMAModel
from unidiffusion.peft.proxy import ProxyNetwork
from unidiffusion.config import LazyConfig
from unidiffusion.utils.checkpoint import save_model_hook, load_model_hook
from unidiffusion.utils.logger import setup_logger
from unidiffusion.utils.snr import snr_loss
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)


class DiffusionTrainer:
    tokenizer = None
    noise_scheduler = None
    text_encoder = None
    vae = None
    unet = None
    unet_ema = None
    weight_dtype = None
    logger = None
    scheduler = None
    lr_scheduler = None
    trainable_params = None
    dataloader = None
    accelerator = None
    optimizer = None
    models = None
    current_iter = 0
    
    def __init__(self, cfg, training):
        self.dataset = None
        self.proxy_model = None
        self.ema_unet = None
        self.cfg = cfg
        self.training = training

        self.default_setup()
        self.build_model()
        self.build_dataloader()
        self.set_placeholders()
        self.load_checkpoint()
        if training:
            self.build_optimizer()
            self.build_scheduler()
            self.build_evaluator()
            self.prepare_training()
            self.print_training_state()
        else:
            self.prepare_inference()

    def default_setup(self):
        # setup log tracker and accelerator
        log_tracker = [platform for platform in ['wandb', 'tensorboard', 'comet_ml'] if self.cfg.train[platform]['enabled']]
        self.cfg.accelerator.log_with = log_tracker[0]  # todo: support multiple loggers
        self.accelerator = instantiate(self.cfg.accelerator)

        if self.accelerator.is_main_process:
            os.makedirs(self.cfg.train.output_dir, exist_ok=True)
            # save all configs
            LazyConfig.save(self.cfg, os.path.join(self.cfg.train.output_dir, 'config.yaml'))

        # prepare checkpoint hook
        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        self.logger = setup_logger(name=__name__, distributed_rank=self.accelerator.process_index)
        os.environ['ACCELERATE_PROCESS_ID'] = str(self.accelerator.process_index)
        self.logger.info(self.accelerator.state)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.train.seed is not None:
            set_seed(self.cfg.train.seed)

        # mixed precision
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def build_dataloader(self):
        self.cfg.dataset.tokenizer = self.tokenizer
        self.dataset = instantiate(self.cfg.dataset)

        self.cfg.dataloader.dataset = self.dataset
        self.dataloader = self.accelerator.prepare(instantiate(self.cfg.dataloader))

    def build_model(self):
        # build proxy model to store trainable modules
        self.proxy_model = ProxyNetwork()
        self.cfg.vae.proxy_model = self.proxy_model
        self.cfg.unet.proxy_model = self.proxy_model
        self.cfg.text_encoder.proxy_model = self.proxy_model

        # build models and noise scheduler
        self.vae = instantiate(self.cfg.vae)
        self.tokenizer = instantiate(self.cfg.tokenizer)
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler)
        self.text_encoder = instantiate(self.cfg.text_encoder)
        self.unet = instantiate(self.cfg.unet)
        self.noise_scheduler = instantiate(self.cfg.noise_scheduler)

        self.models = [self.vae, self.text_encoder, self.unet]
        
        # EMA
        if self.cfg.train.use_ema:
            self.ema_unet = EMAModel(self.unet, model_cls=type(self.unet), model_config=self.unet.config)
        else:
            self.ema_unet = None

        # prepare models
        for model in self.models:
            if model.trainable:
                model = self.accelerator.prepare(model)
            else:
                model.requires_grad_(False)
                model.to(self.accelerator.device, dtype=self.weight_dtype)
        self.proxy_model = self.accelerator.prepare(self.proxy_model)

    def set_placeholders(self):
        placeholders = self.dataset.get_placeholders()
        if placeholders is not None:
            self.tokenizer.set_placeholders(placeholders)
            self.text_encoder.set_placeholders(placeholders, self.tokenizer, self.proxy_model)

    def load_checkpoint(self):
        if self.cfg.train.resume is not None:
            if self.cfg.train.resume != "latest":
                path = os.path.basename(self.cfg.train.resume)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.cfg.train.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.logger.warn(
                    f"Checkpoint '{self.cfg.train.resume}' does not exist. Starting a new training run."
                )
                self.cfg.train.resume = None
            else:
                self.logger.info(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.cfg.train.output_dir, path))
                current_iter = int(path.split("-")[1])
                self.current_iter = current_iter * self.cfg.train.gradient_accumulation_iter

    def build_optimizer(self):
        self.cfg.optimizer.params = self.proxy_model.params_group
        self.optimizer = instantiate(OmegaConf.to_container(self.cfg.optimizer), convert=False)  # not convert list to ListConfig

        # print num of trainable parameters
        num_params = sum([p.numel() for params_group in self.optimizer.param_groups for p in params_group['params']])
        self.logger.info(f"Number of trainable parameters: {num_params / 1e6} M")

    def build_scheduler(self):
        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.cfg.lr_scheduler.num_warmup_steps = self.cfg.train.lr_warmup_iter * self.cfg.train.gradient_accumulation_iter
        self.cfg.lr_scheduler.num_training_steps = self.cfg.train.max_iter * self.cfg.train.gradient_accumulation_iter

        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)

    def build_evaluator(self):
        pass

    def prepare_training(self):
        self.proxy_model.set_requires_grad(True)

        # prepare xformers for unet
        if self.cfg.train.use_xformers and self.unet.trainable:
            self.unet.enable_xformers_memory_efficient_attention()

        # prepare gradient checkpointing
        if self.cfg.train.gradient_checkpointing:
            if self.unet.trainable:
                self.unet.enable_gradient_checkpointing()
            if self.text_encoder.trainable:
                self.text_encoder.gradient_checkpointing_enable()

        # prepare tracker
        output_dir = self.cfg.train.output_dir

        if self.accelerator.is_main_process:
            init_kwargs = dict()
            if self.cfg.train.wandb.enabled:
                wandb_kwargs = {
                    'entity': self.cfg.train.wandb.entity,
                    'name': os.path.split(output_dir)[-1] if os.path.split(output_dir)[-1] != '' else
                    os.path.split(output_dir)[-2],
                }
                init_kwargs['wandb'] = wandb_kwargs
            self.accelerator.init_trackers(self.cfg.train.project, config=vars(self.cfg), init_kwargs=init_kwargs)

    def prepare_inference(self):
        self.proxy_model.set_requires_grad(False)
        # prepare xformers for unet
        if self.cfg.train.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

    def print_training_state(self):
        total_batch_size = self.cfg.dataloader.batch_size * self.accelerator.num_processes * self.cfg.train.gradient_accumulation_iter
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataset)}")
        self.logger.info(f"  Num batches each epoch = {len(self.dataloader)}")
        self.logger.info(f"  Num iterations = {self.cfg.train.max_iter}")
        self.logger.info(f"  Instantaneous batch size per device = {self.cfg.dataloader.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.cfg.train.gradient_accumulation_iter}")
        self.logger.info("****************************")

    def model_train(self):
        for model in self.models:
            if model.trainable:
                model.train()

    def model_eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        self.model_train()

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.cfg.train.max_iter), disable=not self.accelerator.is_local_main_process)
        progress_bar.update(self.current_iter)
        progress_bar.set_description("Steps")

        accelerator, unet, vae, text_encoder, noise_scheduler = self.accelerator, self.unet, self.vae, self.text_encoder, self.noise_scheduler
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        while self.current_iter < self.cfg.train.max_iter:
            batch = next(iter(self.dataloader))
            with accelerator.accumulate(unet):
                if accelerator.is_main_process:
                    # Validation
                    if self.current_iter % self.cfg.inference.inference_iter == 0:
                        self.inference()

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=self.weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if self.cfg.train.snr.enabled:
                    loss = snr_loss(self.cfg.train.snr.snr_gamma, timesteps, noise_scheduler, model_pred, target)
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.proxy_model.parameters(), self.cfg.train.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.current_iter += 1
                    if accelerator.is_main_process:
                        if self.current_iter % self.cfg.train.checkpointing_iter == 0:
                            # Save checkpoint
                            save_path = os.path.join(self.cfg.train.output_dir, f"checkpoint-{self.current_iter:06d}")
                            # Save Unet/VAE/Text Encoder
                            accelerator.save_state(save_path)
                            # Save Tokenizer
                            # ......
                            # Save EMA
                            # ......
                            self.logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=self.current_iter)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save last
            # ......
            pass
        accelerator.end_training()

    def inference(self):
        unet = self.accelerator.unwrap_model(self.unet) if self.unet.trainable else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if self.text_encoder.trainable else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if self.vae.trainable else self.vae

        # create pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.train.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            tokenizer=self.tokenizer,
            unet=unet,
            vae=vae,
            revision=self.cfg.train.revision,
            torch_dtype=self.weight_dtype,
            safety_checker=None,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        if (seed := self.cfg.train.seed) is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(seed)

        # inference prompts
        if self.cfg.inference.prompts is not None:
            prompts = [self.cfg.inference.prompts] * self.cfg.inference.total_num
        else:
            prompts = [self.dataset[index]['prompt'] for index in range(self.cfg.inference.total_num)]

        images = []
        for prompt in prompts:
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    num_inference_steps=self.cfg.inference.num_inference_steps,
                    guidance_scale=self.cfg.inference.guidance_scale,
                    generator=generator
                ).images[0]
            images.append(image)

        # save images
        save_path = os.path.join(self.cfg.train.output_dir, f'visualization-{self.current_iter:06d}')
        os.makedirs(save_path, exist_ok=True)
        for index, image in enumerate(images):
            image_path = os.path.join(save_path, f'img{index + self.accelerator.process_index * self.cfg.inference.total_num:04d}_{prompts[index]}.png')
            image.save(image_path)

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, self.current_iter, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {prompt}") for i, (image, prompt) in
                            enumerate(zip(images, prompts))
                        ]
                    }
                )

        del pipeline
        torch.cuda.empty_cache()

    def evaluate(self):
        pass

