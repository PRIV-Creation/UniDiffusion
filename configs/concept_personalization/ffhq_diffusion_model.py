from configs.common.get_common_config import *
from unidiffusion.config import LazyCall as L
from unidiffusion.models import UNet2DConditionModel_DT


dataset = get_config("common/data/image_dataset.py").dataset

train.project = "UniDiffusion-FFHQ"
train.gradient_checkpointing = False
train.use_ema = True
optimizer.optimizer = '8bit_adam'
optimizer.lr = 1e-5

dataset.path = "datasets/FFHQ/"
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'


train.checkpointing_iter = 10000
dataset.placeholder = 'face'    # not used in null-text mode
dataset.flip_prob = 0.5

# 2 gpus. Total batch size = 2 * 16 * 1 = 32
dataloader.batch_size = 16
train.gradient_accumulation_iter = 1

# Inference
inference.inference_iter = 10000
inference.rectify_uncond = True
inference.pipeline_kwargs.guidance_scale = 2.0
inference.total_num = 80

# Evaluation
evaluation.evaluation_iter = 0

train.output_dir = 'experiments/ffhq/ffhq_null-text_bs32_1e-5'
train.max_iter = 1000000
train.wandb.enabled = True

unet = L(UNet2DConditionModel_DT.from_pretrained)(
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
