from configs.common.get_common_config import *
from unidiffusion.config import LazyCall as L
from unidiffusion.models import UNet2DConditionModel_UniDiffusion


dataset = get_config("common/data/image_dataset.py").dataset

train.project = "UniDiffusion-AFHQ"
train.gradient_checkpointing = False
train.use_ema = True
optimizer.optimizer = '8bit_adam'
optimizer.lr = 1e-5

dataset.path = "datasets/afhq_v2/"
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'


train.checkpointing_iter = 10000
dataset.placeholder = 'afhq'    # not used in null-text mode
dataset.flip_prob = 0.5

# 2 gpus. Total batch size = 2 * 16 * 1 = 32
dataloader.batch_size = 16
train.gradient_accumulation_iter = 1

# Inference
inference.inference_iter = 10000
inference.rectify_uncond = True
inference.guidance_scale = [1.5 + 0.25 * i for i in range(5)]
inference.total_num = 80

# Evaluation
evaluation.evaluation_iter = 10000
evaluation.rectify_uncond = True
evaluation.evaluator.fid.enabled = True
evaluation.evaluator.fid.real_image_num = 1000
evaluation.total_num = 1000
evaluation.guidance_scale = [1.5 + 0.25 * i for i in range(5)]



train.output_dir = 'experiments/afhq/afhq_null-text_bs32_1e-5'
train.max_iter = 1000000
train.wandb.enabled = True

unet = L(UNet2DConditionModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder='unet',
    null_text=True,
    null_text_checkpoint="stable-diffusion-v1-5_null-text.pt",
    training_args = {
        r'': {
            'mode': 'finetune',
            'optim_kwargs': {'lr': '${optimizer.lr}'},
        },
    }
)
