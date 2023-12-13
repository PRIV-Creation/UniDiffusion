from configs.common.get_common_config import *
from unidiffusion.config import LazyCall as L
from unidiffusion.models import UNet2DConditionModel_NullText
from unidiffusion.models import StableDiffusionRectifyPipeline
from unidiffusion.config import LazyCall as L


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
inference.guidance_scale = [1.5 + 0.25 * i for i in range(5)]
inference.total_num = 80

# Evaluation
evaluation.evaluation_iter = 10000
evaluation.rectify_uncond = True
evaluation.evaluator.fid.enabled = True
evaluation.evaluator.fid.real_image_num = 1000
evaluation.total_num = 1000
evaluation.guidance_scale = [1.5 + 0.25 * i for i in range(5)]

train.output_dir = 'experiments/ffhq/ffhq_null-text_bs32_1e-5'
train.max_iter = 1000000
train.wandb.enabled = True

inference_pipeline = L(StableDiffusionRectifyPipeline.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
)

unet = L(UNet2DConditionModel_NullText.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder='unet',
    null_text_checkpoint="stable-diffusion-v1-5_null-text.pt",
    training_args = {
        r'': {
            'mode': 'finetune',
            'optim_kwargs': {'lr': '${optimizer.lr}'},
        },
    }
)
