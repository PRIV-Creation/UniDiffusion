from .unidiffusion_pipeline import UniDiffusionPipeline
from unidiffusion.config import instantiate
from unidiffusion.peft.proxy import ProxyNetwork
from diffusers.training_utils import EMAModel
import shutil
import os
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from unidiffusion.utils.snr import snr_loss
from omegaconf import OmegaConf, ListConfig
from accelerate import PartialState
import torchvision
import numpy as np
import wandb
import random

class SDXL_UniDiffusionTrainingPipeline(UniDiffusionPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mode = "training"
        self.default_setup()
        self.build_model()
        self.prepare_db()
        self.build_dataloader()
        self.prepare_null_text()
        self.set_placeholders() #TODO
        self.build_optimizer()
        self.build_scheduler()
        self.build_evaluator()
        self.prepare_training()
        self.load_checkpoint()
        self.print_training_state()
    def build_model(self):
        self.logger.info("Building model ... ")
        # build proxy model to store trainable modules
        self.proxy_model = ProxyNetwork()
        self.cfg.vae.proxy_model = self.proxy_model
        self.cfg.unet.proxy_model = self.proxy_model
        self.cfg.text_encoder_one.proxy_model = self.proxy_model
        self.cfg.text_encoder_two.proxy_model = self.proxy_model
        # self.cfg.train_batch_size = self.cfg.dataloader.batch_size

        # build models and noise scheduler
        self.vae = instantiate(self.cfg.vae)
        self.tokenizer_one = instantiate(self.cfg.tokenizer_one)
        self.tokenizer_two = instantiate(self.cfg.tokenizer_two)
        self.text_encoder_one = instantiate(self.cfg.text_encoder_one)
        self.text_encoder_two = instantiate(self.cfg.text_encoder_two)
        self.unet = instantiate(self.cfg.unet)
        if not self.cfg.get('only_inference', False):
            self.noise_scheduler = instantiate(self.cfg.noise_scheduler)

        # build origin unet
        self.rectify_uncond = (self.cfg.inference.rectify_uncond or self.cfg.evaluation.rectify_uncond)
        if self.rectify_uncond:
            self.unet_init = instantiate(self.cfg.unet_init)

        self.models = [self.vae, self.text_encoder_one, self.text_encoder_two, self.unet]

        # EMA
        if self.cfg.train.use_ema:
            self.logger.info('Use ema model')
            self.ema_model = EMAModel(self.proxy_model)
        else:
            self.ema_model = None
    def build_dataloader(self):
        self.logger.info("Building dataloader ... ")
        self.cfg.dataset.tokenizer_one = self.tokenizer_one
        self.cfg.dataset.tokenizer_two = self.tokenizer_two
        self.cfg.dataset.text_encoder_one = self.text_encoder_one
        self.cfg.dataset.text_encoder_two = self.text_encoder_two
        self.cfg.dataset.vae = self.vae,
        self.cfg.dataset.accelerator = self.accelerator
        self.cfg.dataset.train_batch_size = self.cfg.dataloader.batch_size
        self.dataset = instantiate(self.cfg.dataset)

        self.cfg.dataloader.dataset = self.dataset
        self.dataloader = self.accelerator.prepare(instantiate(self.cfg.dataloader))
    def set_placeholders(self):
        placeholders = self.dataset.get_placeholders()
        if placeholders is not None:
            self.logger.info(f"Set placeholders:  {placeholders}")
            self.tokenizer_one.set_placeholders(placeholders)
            self.tokenizer_two.set_placeholders(placeholders)
        else:
            self.logger.info("No placeholders found")
        self.text_encoder_one.set_placeholders(placeholders, self.tokenizer_one, self.proxy_model)
        self.text_encoder_two.set_placeholders(placeholders, self.tokenizer_two, self.proxy_model)

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
            # if self.text_encoder_one.trainable: # 原SDXL只对unet.enable_gradient_checkpointing
            #     self.logger.info("Text encoder one enable gradient checkpointing")
            #     self.text_encoder_one.gradient_checkpointing_enable()
            # if self.text_encoder_two.trainable:
            #     self.logger.info("Text encoder two enable gradient checkpointing")
            #     self.text_encoder_two.gradient_checkpointing_enable()

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
        if self.text_encoder_one.trainable:
            self.text_encoder_one = self.accelerator.prepare(self.text_encoder_one)
        else:
            self.text_encoder_one.requires_grad_(False)
            self.text_encoder_one.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.text_encoder_two.trainable:
            self.text_encoder_two = self.accelerator.prepare(self.text_encoder_two)
        else:
            self.text_encoder_two.requires_grad_(False)
            self.text_encoder_two.to(self.accelerator.device, dtype=self.weight_dtype)
        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)
        self.proxy_model = self.accelerator.prepare(self.proxy_model)

        # prepare tracker
        output_dir = self.cfg.train.output_dir

        # save inference and evaluation prompt file
        if self.cfg.inference.prompts is not None and os.path.exists(self.cfg.inference.prompts):
            shutil.copy(self.cfg.inference.prompts, os.path.join(output_dir, 'inference_prompts.txt'))
        if self.cfg.evaluation.prompts is not None and os.path.exists(self.cfg.evaluation.prompts):
            shutil.copy(self.cfg.evaluation.prompts, os.path.join(output_dir, 'evaluation_prompts.txt'))
        if self.cfg.evaluation.evaluator.clip_score.prompts is not None \
                and os.path.exists(self.cfg.evaluation.evaluator.clip_score.prompts):
            shutil.copy(self.cfg.evaluation.evaluator.clip_score.prompts,
                        os.path.join(output_dir, 'evaluation_clip_score_prompts.txt'))
        if self.cfg.evaluation.evaluator.clip_score.prompts_ori is not None \
                and os.path.exists(self.cfg.evaluation.evaluator.clip_score.prompts_ori):
            shutil.copy(self.cfg.evaluation.evaluator.clip_score.prompts_ori,
                        os.path.join(output_dir, 'evaluation_clip_score_prompts_ori.txt'))

    def train(self):
        self.model_train()

        # keep original embeddings as reference

        if (start_token_one_idx := self.accelerator.unwrap_model(self.text_encoder_one).start_token_idx) is not None:
            orig_embeds_one_params = self.accelerator.unwrap_model(self.text_encoder_one).get_input_embeddings().weight.data.clone()
        else:
            orig_embeds_one_params = None
        
        if (start_token_two_idx := self.accelerator.unwrap_model(self.text_encoder_two).start_token_idx) is not None:
            orig_embeds_two_params = self.accelerator.unwrap_model(self.text_encoder_two).get_input_embeddings().weight.data.clone()
        else:
            orig_embeds_two_params = None

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.cfg.train.max_iter), disable=not self.accelerator.is_local_main_process)
        progress_bar.update(self.current_iter)
        progress_bar.set_description("Steps")

        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)

        accelerator, unet, vae, text_encoder_one, text_encoder_two, noise_scheduler = self.accelerator, self.unet, self.vae, self.text_encoder_one, self.text_encoder_two, self.noise_scheduler
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
                    # time_vae_start = time.time()
                    # latents = vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample().detach()
                    # latents = latents * vae.config.scaling_factor
                    # time_vae_end = time.time()
                    latents = batch["model_input"].to(accelerator.device)
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.cfg.train.noise_offset.enabled:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.cfg.train.noise_offset.scale * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    time_text_start = time.time()
                    # encoder_hidden_states_one = text_encoder_one(batch["input_ids"])[-1][-2]
                    # encoder_hidden_states_two = text_encoder_two(batch["input_ids"])[-1][-2]
                    # encoder_hidden_states = torch.concat([encoder_hidden_states_one,encoder_hidden_states_two],dim=-1).to(dtype=self.weight_dtype)
                    # pooled_prompt_embeds = encoder_hidden_states_two[0].to(dtype=self.weight_dtype)
                    def compute_time_ids(original_size, crops_coords_top_left):
                        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                        target_size = self.cfg.train.resolution
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids])
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=self.weight_dtype)
                        return add_time_ids
                    
                    add_time_ids = torch.cat(
                        [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                    )

                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                    time_text_end = time.time()

                    # Predict the noise residual
                    time_unet_start = time.time()
                    model_pred = unet(noisy_latents, timesteps, prompt_embeds,added_cond_kwargs = unet_added_conditions).sample # or [0]?
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
                    if self.accelerator.unwrap_model(self.text_encoder_one).start_token_idx is not None:
                        with torch.no_grad():
                            index_no_updates = torch.ones((len(self.tokenizer_one),), dtype=torch.bool)
                            index_no_updates[start_token_one_idx:] = False
                            accelerator.unwrap_model(text_encoder_one).get_input_embeddings().weight[
                                index_no_updates
                            ] = orig_embeds_one_params[index_no_updates]
                        
                    if self.accelerator.unwrap_model(self.text_encoder_two).start_token_idx is not None:
                        with torch.no_grad():
                            index_no_updates = torch.ones((len(self.tokenizer_two),), dtype=torch.bool)
                            index_no_updates[start_token_two_idx:] = False
                            accelerator.unwrap_model(text_encoder_two).get_input_embeddings().weight[
                                index_no_updates
                            ] = orig_embeds_two_params[index_no_updates]

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
                            # "time/vae": time_vae_end - time_vae_start, 
                            "time/text": time_text_end - time_text_start,
                            "time/unet": time_unet_end - time_unet_start, "time/update": time_update_end - time_update_start}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=self.current_iter)

                time_data_start = time.time()
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
        text_encoder_one = self.accelerator.unwrap_model(self.text_encoder_one) if isinstance(self.text_encoder_one, torch.nn.parallel.DistributedDataParallel) else self.text_encoder_one
        text_encoder_two = self.accelerator.unwrap_model(self.text_encoder_two) if isinstance(self.text_encoder_two, torch.nn.parallel.DistributedDataParallel) else self.text_encoder_two
        vae = self.accelerator.unwrap_model(self.vae) if isinstance(self.vae, torch.nn.parallel.DistributedDataParallel) else self.vae

        # create pipeline
        pipeline_kwargs = {
            'text_encoder_one': text_encoder_one,
            'text_encoder_two': text_encoder_two,
            'tokenizer_one': self.tokenizer_one,
            'tokenizer_two': self.tokenizer_two,
            'unet': unet,
            'vae': vae,
            'revision': self.cfg.train.revision,
            'torch_dtype': self.weight_dtype,
            'safety_checker': None,
        }

        self.cfg.inference_pipeline.update(pipeline_kwargs)
        pipeline = instantiate(self.cfg.inference_pipeline)

        # rectify unconditional guidance (see xxxx)
        if self.cfg.inference.rectify_uncond:
            pipeline.unet_init = self.unet_init.to(dtype=self.weight_dtype, device=self.accelerator.device)

        # set scheduler
        self.cfg.inference.scheduler.config = pipeline.scheduler.config
        pipeline.scheduler = instantiate(self.cfg.inference.scheduler)

        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

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

        if not isinstance(self.cfg.inference.guidance_scale, (ListConfig, list)):
            self.cfg.inference.guidance_scale = [self.cfg.inference.guidance_scale]

        for guidance_scale in self.cfg.inference.guidance_scale:
            if (seed := self.cfg.train.seed) is None:
                generator = None
            else:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(
                    seed + self.accelerator.process_index)

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
                            guidance_scale=guidance_scale,
                            height=self.cfg.inference.resolution[0],
                            width=self.cfg.inference.resolution[1],
                            **self.cfg.inference.forward_kwargs
                        ).images[0]
                        progress_bar.update(1)
                    image_path = os.path.join(save_path,  f'cfg_{guidance_scale:.2f}_img{index + self.accelerator.process_index * self.cfg.inference.total_num:04d}_{prompt}.png')
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
                                    f"validation/cfg{guidance_scale:.2f}": [
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
        text_encoder_one = self.accelerator.unwrap_model(self.text_encoder_one) if isinstance(self.text_encoder_one, torch.nn.parallel.DistributedDataParallel) else self.text_encoder_one
        text_encoder_two = self.accelerator.unwrap_model(self.text_encoder_two) if isinstance(self.text_encoder_two, torch.nn.parallel.DistributedDataParallel) else self.text_encoder_two
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
            'text_encoder_one': text_encoder_one,
            'text_encoder_two': text_encoder_two,
            'tokenizer_one': self.tokenizer_one,
            'tokenizer_two': self.tokenizer_two,
            'unet': unet,
            'vae': vae,
            'revision': self.cfg.train.revision,
            'torch_dtype': self.weight_dtype,
            'safety_checker': None,
        }

        self.cfg.inference_pipeline.update(pipeline_kwargs)
        pipeline = instantiate(self.cfg.inference_pipeline)

        # rectify unconditional guidance (see xxxx)
        if self.cfg.evaluation.rectify_uncond:
            pipeline.unet_init = self.unet_init.to(dtype=self.weight_dtype, device=self.accelerator.device)

        # set scheduler
        self.cfg.evaluation.scheduler.config = pipeline.scheduler.config
        pipeline.scheduler = instantiate(self.cfg.evaluation.scheduler)

        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

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
            random.seed(self.cfg.train.seed)
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

        if not isinstance(self.cfg.evaluation.guidance_scale, (ListConfig, list)):
            self.cfg.evaluation.guidance_scale = [self.cfg.evaluation.guidance_scale]

        results = dict()
        for guidance_scale in self.cfg.evaluation.guidance_scale:
            # run inference
            generator = torch.Generator(device=self.accelerator.device).manual_seed(
                self.cfg.train.seed + self.accelerator.process_index)

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(prompts)), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description("Evaluation Steps")
            for index, prompt in enumerate(prompts):
                with torch.autocast("cuda"):
                    image = pipeline(
                        prompt,
                        generator=generator,
                        output_type='pt',
                        guidance_scale=guidance_scale,
                        height=self.cfg.evaluation.resolution[0],
                        width=self.cfg.evaluation.resolution[1],
                        **self.cfg.evaluation.forward_kwargs
                    ).images[0]
                    progress_bar.update(1)
                    _ = [evaluator.update(
                        image=image[None],
                        text=prompt,                                                  # used for CLIP Score
                        calculate_clip_score=calculate_clip_score_by_general_prompt,  # used for CLIP Score
                        real=False,                                                   # used for FID
                    ) for evaluator in self.evaluators]
                if save_image:
                    image_path = os.path.join(save_path,  f'cfg{guidance_scale:.2f}_img{index + self.accelerator.process_index * self.cfg.evaluation.total_num:04d}_{prompt}.png')
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
                progress_bar.set_description("Evaluation of CLIP Score")
                for index, (prompt, prompt_ori) in enumerate(zip(clip_score_prompts, clip_score_prompts_ori)):
                    with torch.autocast("cuda"):
                        image = pipeline(
                            prompt,
                            generator=generator,
                            output_type='pt',
                            guidance_scale=guidance_scale,
                            height=self.cfg.evaluation.resolution[0],
                            width=self.cfg.evaluation.resolution[1],
                            **self.cfg.evaluation.forward_kwargs
                        ).images[0]

                        _ = [evaluator.update_by_evaluator_name(
                            name="CLIP_Score",
                            image=image[None],
                            text=prompt_ori,                # used for CLIP Score
                            calculate_clip_score=True,      # used for CLIP Score
                        ) for evaluator in self.evaluators]
                        progress_bar.update(1)
                    if save_image:
                        os.makedirs(os.path.join(save_path, f'{prompt_ori}'), exist_ok=True)
                        image_path = os.path.join(save_path,  f'{prompt_ori}', f'cfg{guidance_scale:.2f}_img{index + self.accelerator.process_index * self.cfg.evaluation.total_num:04d}.png')
                        torchvision.transforms.ToPILImage(mode=None)(image).save(image_path)

            self.accelerator.wait_for_everyone()
            tmp_result = dict()
            for evaluator in self.evaluators:
                self.logger.info('Evaluating {} ...'.format(evaluator.name))
                tmp_result.update(evaluator.compute())
                evaluator.reset()
                results[f'cfg{guidance_scale:.2f}'] = tmp_result

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