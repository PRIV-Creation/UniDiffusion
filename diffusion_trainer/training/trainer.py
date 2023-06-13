import diffusers
import os
from diffusion_trainer.config import instantiate
import itertools
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel
from diffusion_trainer.utils.checkpoint import save_model_hook, load_model_hook
from diffusion_trainer.utils.logger import setup_logger
from diffusion_trainer.utils.snr import snr_loss


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
        self.ema_unet = None
        self.cfg = cfg
        self.training = training

        self.default_setup()
        self.build_model()
        self.load_checkpoint()
        if training:
            self.build_dataloader()
            self.build_optimizer()
            self.build_scheduler()
            self.build_evaluator()
            self.build_criterion()
            self.prepare_training()
            self.print_training_state()

    def default_setup(self):
        self.accelerator = instantiate(self.cfg.accelerator)
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

        if self.accelerator.is_main_process:
            os.makedirs(self.cfg.train.output_dir, exist_ok=True)

        # mixed precision
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def build_dataloader(self):
        self.cfg.dataloader.dataset.tokenizer = self.tokenizer
        self.dataloader = self.accelerator.prepare(instantiate(self.cfg.dataloader))

    def build_model(self):
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

        # Load tokenizer
        # .......

    def build_optimizer(self):
        # get trainable parameters
        trainable_params = []
        for model in self.models:
            p = model.get_trainable_params()
            if p is not None:
                trainable_params.append(p)
        trainable_params = itertools.chain(*trainable_params)
        self.trainable_params = trainable_params  # use for grad clip

        # build optimizer
        self.cfg.optimizer.params = trainable_params
        self.optimizer = instantiate(self.cfg.optimizer)

        # print num of trainable parameters
        num_params = sum([p.numel() for params_group in self.optimizer.param_groups for p in params_group ['params']])
        self.logger.info(f"Number of trainable parameters: {num_params / 1e6} M")

    def build_scheduler(self):
        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.cfg.lr_scheduler.num_warmup_steps = self.cfg.train.lr_warmup_iter * self.cfg.train.gradient_accumulation_iter
        self.cfg.lr_scheduler.num_training_steps = self.cfg.train.max_iter * self.cfg.train.gradient_accumulation_iter

        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)

    def build_evaluator(self):
        pass

    def build_criterion(self):
        pass

    def prepare_training(self):
        # prepare models
        for model in self.models:
            if model.trainable:
                model = self.accelerator.prepare(model)
            else:
                model.requires_grad_(False)
                model.to(self.accelerator.device, dtype=self.weight_dtype)

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

        # prepare checkpoint hook
        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def print_training_state(self):
        total_batch_size = self.cfg.dataloader.batch_size * self.accelerator.num_processes * self.cfg.train.gradient_accumulation_iter
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.dataloader.dataset)}")
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
        progress_bar = tqdm(range(self.current_iter, self.cfg.train.max_iter), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        accelerator, unet, vae, text_encoder, noise_scheduler = self.accelerator, self.unet, self.vae, self.text_encoder, self.noise_scheduler
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        while self.current_iter < self.cfg.train.max_iter:
            batch = next(iter(self.dataloader))
            with accelerator.accumulate(unet):
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
                encoder_hidden_states = text_encoder(batch["input_ids"][:, 0])[0].to(dtype=self.weight_dtype)

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
                    accelerator.clip_grad_norm_(self.trainable_params, self.cfg.train.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.current_iter += 1
                    if accelerator.is_main_process:
                        if self.current_iter % self.cfg.train.checkpointing_iter == 0:
                            save_path = os.path.join(self.cfg.train.output_dir, f"checkpoint-{self.current_iter}")
                            accelerator.save_state(save_path)
                            self.logger.info(f"Saved state to {save_path}")

                        # Validation
                        # -------
                        
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
        pass

    def evaluate(self):
        pass

