import diffusers
import os
import numpy as np
import wandb
from unidiffusion.config import instantiate
from tqdm import tqdm
import random
import time
import torchvision
from accelerate import PartialState
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from diffusers import UNet2DConditionModel
from diffusers.training_utils import EMAModel
from unidiffusion.peft.proxy import ProxyNetwork
from unidiffusion.config import LazyConfig
from unidiffusion.evaluation import EVALUATOR
from unidiffusion.utils.checkpoint import save_model_hook, load_model_hook
from unidiffusion.utils.logger import setup_logger
from unidiffusion.utils.snr import snr_loss
from unidiffusion.models.diffusers_pipeline import StableDiffusionUnbiasedPipeline
from diffusers import (
    StableDiffusionPipeline,
)


class UniDiffusionPipeline:
    cfg = None
    mode = None
    tokenizer = None
    noise_scheduler = None
    dataset = None
    text_encoder = None
    proxy_model = None
    vae = None
    unet = None
    unet_init = None
    ema_model = None
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
    evaluators = []
    config = None
    rectify_uncond = False

    def __init__(self, cfg):
        pass

    def default_setup(self):
        # setup log tracker and accelerator
        log_tracker = [platform for platform in ['wandb', 'tensorboard', 'comet_ml'] if self.cfg.train[platform]['enabled']]
        if len(log_tracker) >= 1:
            self.cfg.accelerator.log_with = log_tracker[0]  # todo: support multiple loggers
        self.config = OmegaConf.to_container(self.cfg, resolve=True)
        self.accelerator = instantiate(self.cfg.accelerator)

        if self.accelerator.is_main_process and self.mode == "training":
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
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
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
        self.logger.info("Building dataloader ... ")
        self.cfg.dataset.tokenizer = self.tokenizer
        self.dataset = instantiate(self.cfg.dataset)

        self.cfg.dataloader.dataset = self.dataset
        self.dataloader = self.accelerator.prepare(instantiate(self.cfg.dataloader))

    def build_model(self):
        self.logger.info("Building model ... ")
        # build proxy model to store trainable modules
        self.proxy_model = ProxyNetwork()
        self.cfg.vae.proxy_model = self.proxy_model
        self.cfg.unet.proxy_model = self.proxy_model
        self.cfg.text_encoder.proxy_model = self.proxy_model

        # build models and noise scheduler
        self.vae = instantiate(self.cfg.vae)
        self.tokenizer = instantiate(self.cfg.tokenizer)
        self.text_encoder = instantiate(self.cfg.text_encoder)
        self.unet = instantiate(self.cfg.unet)
        if not self.cfg.get('only_inference', False):
            self.noise_scheduler = instantiate(self.cfg.noise_scheduler)

        # build origin unet
        self.rectify_uncond = (self.cfg.inference.rectify_uncond or self.cfg.evaluation.rectify_uncond)
        if self.rectify_uncond:
            self.unet_init = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path=self.cfg.train.pretrained_model_name_or_path,
                subfolder="unet",
            )

        self.models = [self.vae, self.text_encoder, self.unet]

        # EMA
        if self.cfg.train.use_ema:
            self.logger.info('Use ema model')
            self.ema_model = EMAModel(self.proxy_model)
        else:
            self.ema_model = None

    def set_placeholders(self):
        placeholders = self.dataset.get_placeholders()
        if placeholders is not None:
            self.logger.info(f"Set placeholders:  {placeholders}")
            self.tokenizer.set_placeholders(placeholders)
        else:
            self.logger.info("No placeholders found")
        self.text_encoder.set_placeholders(placeholders, self.tokenizer, self.proxy_model)

    def load_checkpoint(self):
        if self.cfg.train.resume is not None:
            if self.cfg.train.resume != "latest":
                path = os.path.split(self.cfg.train.resume)
                path = os.path.split(path[0])[-1] if path[-1] == "" else path[-1]
                checkpoint_path = self.cfg.train.resume
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.cfg.train.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                checkpoint_path = os.path.join(self.cfg.train.output_dir, path)

            if path is None:
                self.logger.warn(
                    f"Checkpoint '{self.cfg.train.resume}' does not exist. Starting a new pipelines run."
                )
                self.cfg.train.resume = None
            else:
                self.logger.info(f"Resuming from checkpoint {path}")
                if not self.cfg.checkpoint.load_optimizer:
                    scaler, self.accelerator.scaler = self.accelerator.scaler, None
                    self.accelerator.load_state(checkpoint_path)
                    self.accelerator.scaler = scaler
                else:
                    self.accelerator.load_state(checkpoint_path)

                if self.ema_model is not None:
                    if os.path.exists(ema_model_path := os.path.join(checkpoint_path, 'ema.pt')):
                        self.ema_model.load_state_dict(torch.load(ema_model_path, map_location='cpu'))
                    else:
                        self.logger.info(f"EMA model checkpoint doesn't exist from checkpoint {path}!")
                self.current_iter = int(path.split("-")[1])
        else:
            self.logger.info("Starting a new pipelines run.")

        # prepare optimizer and scheduler if they are not resumed.
        if not self.cfg.checkpoint.load_optimizer:
            self.optimizer = self.accelerator.prepare(self.optimizer)
        if not self.cfg.checkpoint.load_scheduler:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def build_optimizer(self):
        self.logger.info("Building optimizer ... ")
        self.cfg.optimizer.params = self.proxy_model.params_group
        self.optimizer = instantiate(OmegaConf.to_container(self.cfg.optimizer), convert=False)  # not convert list to ListConfig

        if self.cfg.checkpoint.load_optimizer:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # print num of trainable parameters
        num_params = sum([p.numel() for params_group in self.optimizer.param_groups for p in params_group['params']])
        self.logger.info(f"Number of trainable parameters: {num_params / 1e6} M")

    def build_scheduler(self):
        self.logger.info("Building scheduler ... ")
        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.cfg.lr_scheduler.num_warmup_steps = self.cfg.train.lr_warmup_iter * self.cfg.train.gradient_accumulation_iter
        self.cfg.lr_scheduler.num_training_steps = self.cfg.train.max_iter * self.cfg.train.gradient_accumulation_iter

        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)

        if self.cfg.checkpoint.load_scheduler:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def build_evaluator(self):
        self.logger.info("Building evaluator ... ")
        for evaluator, evaluator_args in self.cfg.evaluation.evaluator.items():
            if evaluator_args.get('enabled'):
                self.evaluators.append(EVALUATOR[evaluator](**evaluator_args).to(self.accelerator.device))
                self.logger.info(f'Build {evaluator} evaluator.')
        # _ = [evaluator.to(self.accelerator.device) for evaluator in self.evaluators]

    def prepare_null_text(self):
        if not self.cfg.train.null_text:
            return None
        # prepare null_text


    def prepare_db(self):
        if 'db' not in self.cfg.train:
            return None

        db_cfg = self.cfg.train.db
        if not db_cfg.with_prior_preservation:
            if db_cfg.class_data_dir is not None:
                self.logger.warn("You need not use --class_data_dir without --with_prior_preservation.")
            if db_cfg.class_prompt is not None:
                self.logger.warn("You need not use --class_prompt without --with_prior_preservation.")
            return

        if db_cfg.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if db_cfg.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")   
        # Generate class images if prior preservation is enabled.
        class_images_dir = Path(db_cfg.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < db_cfg.num_class_images:
            torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
            if db_cfg.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif db_cfg.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif db_cfg.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = db_cfg.num_class_images - cur_class_images
            self.logger.info(f"Number of class images to sample: {num_new_images}.")
            class PromptDataset(Dataset):
                "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

                def __init__(self, prompt, num_samples):
                    self.prompt = prompt
                    self.num_samples = num_samples

                def __len__(self):
                    return self.num_samples

                def __getitem__(self, index):
                    example = {}
                    example["prompt"] = self.prompt
                    example["index"] = index
                    return example
            sample_dataset = PromptDataset(db_cfg.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)

            sample_dataloader = self.accelerator.prepare(sample_dataloader)
            pipeline.to(self.accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not self.accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_training(self):
        self.logger.info("Preparing training ... ")
        # prepare proxy model
        self.proxy_model.set_requires_grad(True)

        # prepare xformers for unet
        if self.cfg.train.use_xformers and self.unet.trainable:
            self.unet.enable_xformers_memory_efficient_attention()

        # prepare gradient checkpointing
        if self.cfg.train.gradient_checkpointing:
            if self.unet.trainable:
                self.logger.info("Unet enable gradient checkpointing")
                self.unet.enable_gradient_checkpointing()
            if self.text_encoder.trainable:
                self.logger.info("Text encoder enable gradient checkpointing")
                self.text_encoder.gradient_checkpointing_enable()

        # prepare evaluator
        for evaluator in self.evaluators:
            evaluator.before_train(self.dataset, self.accelerator)
            self.logger.info(evaluator)

        # prepare models
        if self.unet.trainable:
            self.unet = self.accelerator.prepare(self.unet)
        else:
            self.unet.requires_grad_(False)
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.vae.trainable:
            self.vae = self.accelerator.prepare(self.vae)
        else:
            self.vae.requires_grad_(False)
            self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.text_encoder.trainable:
            self.text_encoder = self.accelerator.prepare(self.text_encoder)
        else:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)
        self.proxy_model = self.accelerator.prepare(self.proxy_model)

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
            self.accelerator.init_trackers(self.cfg.train.project, config=self.config, init_kwargs=init_kwargs)
            if self.cfg.train.wandb.enabled:
                wandb.define_metric("inference/step")
                wandb.define_metric("inference/*", step_metric="inference/step")
        else:
            self.accelerator.init_trackers(self.cfg.train.project)

    def prepare_inference(self, prepare_evaluator=False):
        # prepare models
        self.unet.requires_grad_(False)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.requires_grad_(False)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.ema_model.to(self.accelerator.device)
        self.proxy_model = self.accelerator.prepare(self.proxy_model)
        self.proxy_model.set_requires_grad(False)
        # prepare xformers for unet
        if self.cfg.train.use_xformers:
            self.unet.enable_xformers_memory_efficient_attention()
        # prepare evaluator
        if prepare_evaluator:
            for evaluator in self.evaluators:
                evaluator.before_train(self.dataset, self.accelerator)
                self.logger.info(evaluator)

    def print_training_state(self):
        total_batch_size = self.cfg.dataloader.batch_size * self.accelerator.num_processes * self.cfg.train.gradient_accumulation_iter
        self.logger.info("***** Running pipelines *****")
        self.logger.info(f"  Num examples = {len(self.dataset)}")
        self.logger.info(f"  Num batches each epoch = {len(self.dataloader)}")
        self.logger.info(f"  Num iterations = {self.cfg.train.max_iter}")
        self.logger.info(f"  Instantaneous batch size per device = {self.cfg.dataloader.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.cfg.train.gradient_accumulation_iter}")
        self.logger.info("****************************")

    def model_train(self):
        for model in self.models:
            try:
                if model.trainable:
                    model.train()
            except:
                if model.module.trainable:
                    model.train()

    def train(self):
        self.model_train()

        # keep original embeddings as reference

        if (start_token_idx := self.accelerator.unwrap_model(self.text_encoder).start_token_idx) is not None:
            orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()
        else:
            orig_embeds_params = None

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.cfg.train.max_iter), disable=not self.accelerator.is_local_main_process)
        progress_bar.update(self.current_iter)
        progress_bar.set_description("Steps")

        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)

        accelerator, unet, vae, text_encoder, noise_scheduler = self.accelerator, self.unet, self.vae, self.text_encoder, self.noise_scheduler
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        accumulation_first_iter = True
        while self.current_iter < self.cfg.train.max_iter:
            time_data_start = time.time()
            for batch in self.dataloader:
                time_data_end = time.time()
                if self.current_iter >= self.cfg.train.max_iter:
                    break
                with accelerator.accumulate(self.proxy_model):
                    # ------------------------------------------------------------
                    # 1. Inference
                    # ------------------------------------------------------------
                    if self.cfg.inference.inference_iter > 0 \
                            and self.current_iter % self.cfg.inference.inference_iter == 0 and accumulation_first_iter:
                        if self.cfg.inference.skip_error:
                            try:
                                self.inference()
                            except:
                                pass
                        else:
                            self.inference()
                    # ------------------------------------------------------------
                    # 2. Evaluation
                    # ------------------------------------------------------------
                    if self.cfg.evaluation.evaluation_iter > 0 \
                            and len(self.evaluators) >= 1 \
                            and self.current_iter % self.cfg.evaluation.evaluation_iter == 0 \
                            and accumulation_first_iter:
                        if self.cfg.evaluation.skip_error:
                            try:
                                evaluation_results = self.evaluate()
                                evaluation_results["step"] = self.current_iter
                                accelerator.log({'inference/' + k: v for k, v in evaluation_results.items()})
                            except:
                                pass
                        else:
                            evaluation_results = self.evaluate()
                            evaluation_results["step"] = self.current_iter
                            accelerator.log({'inference/' + k: v for k, v in evaluation_results.items()})
                    accumulation_first_iter = False
                    # ------------------------------------------------------------
                    # 3. Diffusion and Denoising
                    # ------------------------------------------------------------
                    # Convert images to latent space
                    time_vae_start = time.time()
                    latents = vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor
                    time_vae_end = time.time()

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
                    time_text_start = time.time()
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=self.weight_dtype)
                    time_text_end = time.time()

                    # Predict the noise residual
                    time_unet_start = time.time()
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    time_unet_end = time.time()

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # ------------------------------------------------------------
                    # 5. Calculate loss and backward
                    # ------------------------------------------------------------
                    if 'db' in self.cfg.train and self.cfg.train.db.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.cfg.train.db.prior_loss_weight * prior_loss
                    else:
                        if self.cfg.train.snr.enabled:
                            loss = snr_loss(self.cfg.train.snr.snr_gamma, timesteps, noise_scheduler, model_pred, target)
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    time_update_start = time.time()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.proxy_model.parameters(), self.cfg.train.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    time_update_end = time.time()

                    # ------------------------------------------------------------
                    # 6. Keep not trainable text embedding unchanged
                    # ------------------------------------------------------------
                    if self.accelerator.unwrap_model(self.text_encoder).start_token_idx is not None:
                        with torch.no_grad():
                            index_no_updates = torch.ones((len(self.tokenizer),), dtype=torch.bool)
                            index_no_updates[start_token_idx:] = False
                            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                                index_no_updates
                            ] = orig_embeds_params[index_no_updates]

                    if accelerator.sync_gradients:
                        # --------------------------------------------------------
                        # 7. Update ema
                        # --------------------------------------------------------
                        if self.ema_model is not None:
                            self.ema_model.step(self.proxy_model.parameters())
                        # --------------------------------------------------------
                        # 8. Save checkpoint
                        # --------------------------------------------------------
                        if accelerator.is_main_process:
                            # Save checkpoint
                            if self.current_iter % self.cfg.train.checkpointing_iter == 0 or \
                                    self.current_iter == (self.cfg.train.max_iter - 1):
                                save_path = os.path.join(self.cfg.train.output_dir, f"checkpoint-{self.current_iter:06d}")
                                # Save Unet/VAE/Text Encoder
                                accelerator.save_state(save_path)
                                # Save EMA model
                                if self.ema_model is not None:
                                    torch.save(self.ema_model.state_dict(), os.path.join(save_path, 'ema.pt'))
                                self.logger.info(f"Saved state to {save_path}")

                        progress_bar.update(1)
                        self.current_iter += 1
                        accumulation_first_iter = True

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                            "time/data": time_data_end - time_data_start,
                            "time/vae": time_vae_end - time_vae_start, "time/text": time_text_end - time_text_start,
                            "time/unet": time_unet_end - time_unet_start, "time/update": time_update_end - time_update_start}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=self.current_iter)
        accelerator.end_training()

    def inference(self):
        self.logger.info('Start inference ... ')

        # prepare inference models and pipeline
        if self.ema_model is not None:
            self.ema_model.store(self.proxy_model.parameters())
            self.ema_model.copy_to(self.proxy_model.parameters())
            self.ema_model.to("cpu") # to save GPU memory
        torch.cuda.empty_cache()

        unet = self.accelerator.unwrap_model(self.unet) if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae, torch.nn.parallel.DistributedDataParallel) else self.vae

        # create pipeline
        pipeline_kwargs = {
            'text_encoder': text_encoder,
            'tokenizer': self.tokenizer,
            'unet': unet,
            'vae': vae,
            'revision': self.cfg.train.revision,
            'torch_dtype': self.weight_dtype,
            'safety_checker': None,
        }

        # rectify unconditional guidance (see xxxx)
        if self.cfg.inference.rectify_uncond:
            pipeline = StableDiffusionUnbiasedPipeline.from_pretrained(
                self.cfg.train.pretrained_model_name_or_path,
                **pipeline_kwargs
            )
            pipeline.unet_init = self.unet_init.to(dtype=self.weight_dtype, device=self.accelerator.device)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfg.train.pretrained_model_name_or_path,
                **pipeline_kwargs
            )

        # set scheduler
        pipeline.scheduler = diffusers.__dict__[self.cfg.inference.scheduler].from_config(pipeline.scheduler.config)

        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if (seed := self.cfg.train.seed) is None:
            generator = None
        else:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(seed + self.accelerator.process_index)

        # inference prompts
        if self.cfg.inference.prompts is not None:
            if os.path.isfile(self.cfg.inference.prompts):
                with open(self.cfg.inference.prompts, 'r') as f:
                    prompts = [line.strip() for line in f.readlines()]
                    prompts = [p for p in prompts if p != ""]
                    prompts = (prompts * (self.cfg.inference.total_num // len(prompts) + 1))[:self.cfg.inference.total_num]
                    prompts.sort()
            else:
                prompts = [self.cfg.inference.prompts] * self.cfg.inference.total_num
        else:
            prompts = [self.dataset[index]['prompt'] for index in range(self.cfg.inference.total_num)]

        # save images
        if (save_path := self.cfg.inference.save_path) is None:
            save_path = os.path.join(self.cfg.train.output_dir, f'visualization-{self.current_iter:06d}')
        os.makedirs(save_path, exist_ok=True)

        distributed_state = PartialState()
        with distributed_state.split_between_processes(prompts) as prompt_per_card:
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(prompt_per_card)), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description("Steps")
            images = []
            for index, prompt in enumerate(prompt_per_card):
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        generator=generator,
                        output_type='pt',
                        **self.cfg.inference.pipeline_kwargs
                    ).images[0]
                    progress_bar.update(1)
                image_path = os.path.join(save_path,  f'img{index + self.accelerator.process_index * self.cfg.inference.total_num:04d}_{prompt}.png')
                torchvision.transforms.ToPILImage(mode=None)(image).save(image_path)
                if self.mode == "training":
                    images.append(image)

        if self.mode == "training":
            images = torch.stack(images)
            images = self.accelerator.gather(images)
            if self.accelerator.is_main_process:
                for tracker in self.accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.asarray(images)
                        tracker.writer.add_images("inference", np_images, self.current_iter, dataformats="NHWC")
                    if tracker.name == "wandb":
                        self.logger.info("Logging images to wandb")
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {prompt}") for i, (image, prompt) in
                                    enumerate(zip(images, prompts))
                                ]
                            }
                        )
                        self.logger.info("Logging images to wandb finish!")

            if self.ema_model is not None:
                self.ema_model.restore(self.proxy_model.parameters())
                self.ema_model.to(self.accelerator.device)

        self.accelerator.wait_for_everyone()
        del images
        del pipeline
        torch.cuda.empty_cache()

    def evaluate(self):
        torch.cuda.empty_cache()
        self.logger.info('Start evaluation ... ')

        # Load all evaluator to device
        _ = [evaluator.to(self.accelerator.device) for evaluator in self.evaluators]

        if self.ema_model is not None:
            self.ema_model.store(self.proxy_model.parameters())
            self.ema_model.copy_to(self.proxy_model.parameters())
            self.ema_model.to("cpu") # to save GPU memory

        unet = self.accelerator.unwrap_model(self.unet) if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae,  torch.nn.parallel.DistributedDataParallel) else self.vae

        # save images
        if self.cfg.evaluation.save_image:
            save_image = True
            if (save_path := self.cfg.evaluation.save_path) is None:
                save_path = os.path.join(self.cfg.train.output_dir, f'evaluation-{self.current_iter:06d}')
            os.makedirs(save_path, exist_ok=True)
        else:
            save_image = False
            save_path = None

        # create pipeline
        pipeline_kwargs = {
            'text_encoder': text_encoder,
            'tokenizer': self.tokenizer,
            'unet': unet,
            'vae': vae,
            'revision': self.cfg.train.revision,
            'torch_dtype': self.weight_dtype,
            'safety_checker': None,
        }
        if self.cfg.evaluation.rectify_uncond:
            pipeline = StableDiffusionUnbiasedPipeline.from_pretrained(
                self.cfg.train.pretrained_model_name_or_path,
                **pipeline_kwargs
            )
            pipeline.unet_init = self.unet_init.to(dtype=self.weight_dtype, device=self.accelerator.device)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.cfg.train.pretrained_model_name_or_path,
                **pipeline_kwargs
            )
        pipeline.scheduler = diffusers.__dict__[self.cfg.inference.scheduler].from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.cfg.train.seed + self.accelerator.process_index)

        # random select different data for each process
        if self.cfg.evaluation.prompts is not None:
            if os.path.isfile(self.cfg.evaluation.prompts):
                with open(self.cfg.evaluation.prompts, 'r') as f:
                    prompts = [line.strip() for line in f.readlines()]
                    prompts = [p for p in prompts if p != ""]
                prompts = (prompts * (self.cfg.evaluation.total_num // len(prompts) + 1))[:self.cfg.evaluation.total_num]
                prompts.sort()
            else:
                prompts = [self.cfg.evaluation.prompts] * (self.cfg.evaluation.total_num // self.accelerator.num_processes)
        else:
            # to keep the same prompts with real images and different between processes
            random.seed(0)
            total_idx = random.sample(list(range(len(self.dataset))) * (int(self.cfg.evaluation.total_num / len(self.dataset)) + 1), self.cfg.evaluation.total_num)
            image_per_process = len(total_idx) // self.accelerator.num_processes
            process_idx = total_idx[self.accelerator.process_index * image_per_process: (self.accelerator.process_index + 1) * image_per_process]
            prompts = [self.dataset.get_prompt(index) for index in process_idx]

        # If no prompts are specified for clip score, use the evaluation prompts.
        if self.cfg.evaluation.evaluator.clip_score.prompts is not None \
                and self.cfg.evaluation.evaluator.clip_score.enabled:
            calculate_clip_score_by_general_prompt = False
            with open(self.cfg.evaluation.evaluator.clip_score.prompts, 'r') as f:
                clip_score_prompts = [line.strip() for line in f.readlines()]
                clip_score_prompts = [p for p in clip_score_prompts if p != ""]
                clip_score_prompts = (clip_score_prompts * (self.cfg.evaluation.evaluator.clip_score.total_num // (len(clip_score_prompts) * self.accelerator.num_processes) + 1))[:self.cfg.evaluation.evaluator.clip_score.total_num // self.accelerator.num_processes]
        else:
            calculate_clip_score_by_general_prompt = True

        if self.cfg.evaluation.evaluator.clip_score.prompts_ori is not None \
                and self.cfg.evaluation.evaluator.clip_score.enabled:
            with open(self.cfg.evaluation.evaluator.clip_score.prompts_ori, 'r') as f:
                clip_score_prompts_ori = [line.strip() for line in f.readlines()]
                clip_score_prompts_ori = [p for p in clip_score_prompts_ori if p != ""]
                clip_score_prompts_ori = (clip_score_prompts_ori * (self.cfg.evaluation.evaluator.clip_score.total_num // (len(clip_score_prompts_ori) * self.accelerator.num_processes) + 1))[:self.cfg.evaluation.evaluator.clip_score.total_num // self.accelerator.num_processes]

        # GCFG
        guidance_scale_ori = self.cfg.evaluation.pipeline_kwargs.pop("guidance_scale_ori", None)
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(prompts)), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for index, prompt in enumerate(prompts):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    generator=generator,
                    output_type='pt',
                    **self.cfg.evaluation.pipeline_kwargs
                ).images[0]
                progress_bar.update(1)
                _ = [evaluator.update(
                    image=image[None],
                    text=prompt,                                                  # used for CLIP Score
                    calculate_clip_score=calculate_clip_score_by_general_prompt,  # used for CLIP Score
                    real=False,                                                   # used for FID
                ) for evaluator in self.evaluators]
            if save_image:
                image_path = os.path.join(save_path,  f'img{index + self.accelerator.process_index * self.cfg.evaluation.total_num:04d}_{prompt}.png')
                torchvision.transforms.ToPILImage(mode=None)(image).save(image_path)

        # Evaluate CLIP Score
        if not calculate_clip_score_by_general_prompt:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(
                self.cfg.train.seed + self.accelerator.process_index)
            progress_bar = tqdm(
                range(len(clip_score_prompts)),
                desc="Evaluating CLIP Score",
                disable=not self.accelerator.is_local_main_process
            )
            progress_bar.set_description("Steps")
            for index, (prompt, prompt_ori) in enumerate(zip(clip_score_prompts, clip_score_prompts_ori)):
                with torch.autocast("cuda"):
                    image = pipeline(
                        [prompt, prompt_ori],
                        generator=generator,
                        output_type='pt',
                        guidance_scale_ori=guidance_scale_ori,
                        **self.cfg.evaluation.pipeline_kwargs
                    ).images[0]

                    _ = [evaluator.update_by_evaluator_name(
                        name="CLIP_Score",
                        image=image[None],
                        text=prompt,                    # used for CLIP Score
                        calculate_clip_score=True,      # used for CLIP Score
                    ) for evaluator in self.evaluators]
                    progress_bar.update(1)
                if save_image:
                    os.makedirs(os.path.join(save_path, f'{prompt_ori}'), exist_ok=True)
                    image_path = os.path.join(save_path,  f'{prompt_ori}', f'img{index + self.accelerator.process_index * self.cfg.evaluation.total_num:04d}.png')
                    torchvision.transforms.ToPILImage(mode=None)(image).save(image_path)

        self.accelerator.wait_for_everyone()
        results = dict()
        for evaluator in self.evaluators:
            self.logger.info('Evaluating {} ...'.format(evaluator.name))
            results.update(evaluator.compute())
            evaluator.reset()
        if self.ema_model is not None:
            self.ema_model.restore(self.proxy_model.parameters())
            self.ema_model.to(self.accelerator.device)

        # release evaluator memory
        _ = [evaluator.to("cpu") for evaluator in self.evaluators]

        del image
        del pipeline
        torch.cuda.empty_cache()

        self.logger.info(f'Evaluation results:\n{results}')

        return results

    def save_diffusers(self, save_path):
        if self.ema_model is not None:
            self.ema_model.store(self.proxy_model.parameters())
            self.ema_model.copy_to(self.proxy_model.parameters())
            self.ema_model.to("cpu")

        unet = self.accelerator.unwrap_model(self.unet) if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae,  torch.nn.parallel.DistributedDataParallel) else self.vae

        # create pipeline
        pipeline_kwargs = {
            'text_encoder': text_encoder,
            'tokenizer': self.tokenizer,
            'unet': unet,
            'vae': vae,
            'revision': self.cfg.train.revision,
            'torch_dtype': self.weight_dtype,
            'safety_checker': None,
        }
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.train.pretrained_model_name_or_path,
            **pipeline_kwargs
        )
        pipeline.save_pretrained(save_path)
