import warnings
from pathlib import Path
import hashlib
import diffusers
from diffusers import DiffusionPipeline
import os
import numpy as np
import wandb
from unidiffusion.config import instantiate
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import transformers
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from diffusers.training_utils import EMAModel
from unidiffusion.peft.proxy import ProxyNetwork
from unidiffusion.config import LazyConfig
from unidiffusion.evaluation import EVALUATOR
from unidiffusion.utils.checkpoint import save_model_hook, load_model_hook
from unidiffusion.utils.logger import setup_logger
from unidiffusion.utils.snr import snr_loss
from diffusers import (
    StableDiffusionPipeline,
)


class UniDiffusionPipeline:
    tokenizer = None
    noise_scheduler = None
    dataset = None
    text_encoder = None
    proxy_model = None
    vae = None
    unet = None
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

    def __init__(self, cfg, training):
        self.cfg = cfg
        self.training = training
        self.default_setup()
        self.build_model()
        if training:
            self.prepare_db()
            self.build_dataloader()
            self.set_placeholders()

        if training:
            self.build_optimizer()
            self.build_scheduler()
            self.build_evaluator()
            self.prepare_training()
            self.print_training_state()
        else:
            self.prepare_inference()

        self.load_checkpoint()

    def default_setup(self):
        # setup log tracker and accelerator
        log_tracker = [platform for platform in ['wandb', 'tensorboard', 'comet_ml'] if self.cfg.train[platform]['enabled']]
        if len(log_tracker) >= 1:
            self.cfg.accelerator.log_with = log_tracker[0]  # todo: support multiple loggers
        self.config = OmegaConf.to_container(self.cfg, resolve=True)
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
            self.text_encoder.set_placeholders(placeholders, self.tokenizer, self.proxy_model)
        else:
            self.logger.info("No placeholders found")

    def load_checkpoint(self):
        if self.cfg.train.resume is not None:
            if self.cfg.train.resume != "latest":
                path = os.path.basename(self.cfg.train.resume)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.cfg.train.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.logger.warn(
                    f"Checkpoint '{self.cfg.train.resume}' does not exist. Starting a new pipelines run."
                )
                self.cfg.train.resume = None
            else:
                self.logger.info(f"Resuming from checkpoint {path}")
                checkpoint_path = os.path.join(self.cfg.train.output_dir, path)
                self.accelerator.load_state(checkpoint_path)
                if self.ema_model is not None:
                    self.ema_model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'ema.pt'), map_location='cpu'))
                self.current_iter = int(path.split("-")[1])
        else:
            self.logger.info("Starting a new pipelines run.")

    def build_optimizer(self):
        self.logger.info("Building optimizer ... ")
        self.cfg.optimizer.params = self.proxy_model.params_group
        self.optimizer = instantiate(OmegaConf.to_container(self.cfg.optimizer), convert=False)  # not convert list to ListConfig
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
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

    def build_evaluator(self):
        self.logger.info("Building evaluator ... ")
        for evaluator, evaluator_args in self.cfg.evaluation.evaluator.items():
            if evaluator_args.pop('enabled'):
                self.evaluators.append(EVALUATOR[evaluator](**evaluator_args).to(self.accelerator.device))
                self.logger.info(f'Build {evaluator} evaluator.')
        _ = [evaluator.to(self.accelerator.device) for evaluator in self.evaluators]

    def prepare_db(self):
        if 'db' not in self.cfg.train:
            return None

        db_cfg = self.cfg.train.db
        if not db_cfg.with_prior_preservation:
            if db_cfg.class_data_dir is not None:
                warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
            if db_cfg.class_prompt is not None:
                warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
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

    def prepare_inference(self):
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
        if (start_token_idx := self.text_encoder.start_token_idx) is not None:
            orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()
        else:
            orig_embeds_params = None

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.cfg.train.max_iter), disable=not self.accelerator.is_local_main_process)
        progress_bar.update(self.current_iter)
        progress_bar.set_description("Steps")

        accelerator, unet, vae, text_encoder, noise_scheduler = self.accelerator, self.unet, self.vae, self.text_encoder, self.noise_scheduler
        optimizer, lr_scheduler = self.optimizer, self.lr_scheduler
        while self.current_iter < self.cfg.train.max_iter:
            batch = next(iter(self.dataloader))
            with accelerator.accumulate(self.proxy_model):
                # ------------------------------------------------------------
                # 1. Inference
                # ------------------------------------------------------------
                if accelerator.is_main_process:
                    # Validation
                    if self.cfg.inference.inference_iter > 0 and \
                            self.current_iter % self.cfg.inference.inference_iter == 0:
                        self.inference()
                # ------------------------------------------------------------
                # 2. Evaluation
                # ------------------------------------------------------------
                if len(self.evaluators) >= 1 and self.current_iter % self.cfg.evaluation.evaluation_iter == 0:
                    evaluation_results = self.evaluate()
                    accelerator.log(evaluation_results, step=self.current_iter)

                # ------------------------------------------------------------
                # 3. Diffusion and Denoising
                # ------------------------------------------------------------
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

                # ------------------------------------------------------------
                # 5. Calculate loss and backward
                # ------------------------------------------------------------
                if self.cfg.train.db.with_prior_preservation:
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

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.proxy_model.parameters(), self.cfg.train.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # ------------------------------------------------------------
                # 6. Keep not trainable text embedding unchanged
                # ------------------------------------------------------------
                if self.text_encoder.start_token_idx is not None:
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
                        if self.current_iter % self.cfg.train.checkpointing_iter == 0:
                            save_path = os.path.join(self.cfg.train.output_dir, f"checkpoint-{self.current_iter:06d}")
                            # Save Unet/VAE/Text Encoder
                            accelerator.save_state(save_path)
                            # Save EMA model
                            if self.ema_model is not None:
                                torch.save(self.ema_model.state_dict(), os.path.join(save_path, 'ema.pt'))
                            self.logger.info(f"Saved state to {save_path}")

                    progress_bar.update(1)
                    self.current_iter += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=self.current_iter)
        accelerator.end_training()

    def inference(self):
        self.logger.info('Start inference ... ')
        if self.ema_model is not None:
            self.ema_model.store(self.proxy_model.parameters())
            self.ema_model.copy_to(self.proxy_model.parameters())

        unet = self.accelerator.unwrap_model(self.unet) if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae, torch.nn.parallel.DistributedDataParallel) else self.vae

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
        pipeline.scheduler = diffusers.__dict__[self.cfg.inference.scheduler].from_config(pipeline.scheduler.config)
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

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(prompts)), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        images = []
        for prompt in prompts:
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    generator=generator,
                    **self.cfg.inference.pipeline_kwargs
                ).images[0]
                progress_bar.update(1)
            images.append(image)

        # save images
        if (save_path := self.cfg.inference.save_path) is None:
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

        if self.ema_model is not None:
            self.ema_model.restore(self.proxy_model.parameters())

        del pipeline
        torch.cuda.empty_cache()

    def evaluate(self):
        self.logger.info('Start evaluation ... ')
        if self.ema_model is not None:
            self.ema_model.store(self.proxy_model.parameters())
            self.ema_model.copy_to(self.proxy_model.parameters())
        unet = self.accelerator.unwrap_model(self.unet) if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
        text_encoder = self.accelerator.unwrap_model(self.text_encoder) if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae,  torch.nn.parallel.DistributedDataParallel) else self.vae

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
        pipeline.scheduler = diffusers.__dict__[self.cfg.inference.scheduler].from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.cfg.train.seed + self.accelerator.process_index)

        # random select different data for each process
        if self.cfg.evaluation.prompts is not None:
            prompts = [self.cfg.evaluation.prompts] * self.cfg.evaluation.total_num
        else:
            # to keep the same prompts with real images and different between processes
            random.seed(0)
            total_idx = random.sample(range(len(self.dataset)), min(self.cfg.evaluation.total_num, len(self.dataset)))
            image_per_process = len(total_idx) // self.accelerator.num_processes
            process_idx = total_idx[self.accelerator.process_index * image_per_process: (self.accelerator.process_index + 1) * image_per_process]
            prompts = [self.dataset[index]['prompt'] for index in process_idx]

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(prompts)), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        for prompt in prompts:
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt,
                    generator=generator,
                    output_type='pt',
                    **self.cfg.evaluation.pipeline_kwargs
                ).images[0]
                _ = [evaluator.update(image[None], real=False) for evaluator in self.evaluators]
                progress_bar.update(1)

        results = dict()
        for evaluator in self.evaluators:
            self.logger.info('Evaluating {} ...'.format(evaluator.name))
            results.update(evaluator.compute())
            evaluator.reset()

        if self.ema_model is not None:
            self.ema_model.restore(self.proxy_model.parameters())

        del pipeline
        torch.cuda.empty_cache()

        self.logger.info(f'Evaluation results:\n{results}')
        return results
