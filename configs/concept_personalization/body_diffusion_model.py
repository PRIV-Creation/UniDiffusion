from configs.common.get_common_config import *
from unidiffusion.config import LazyCall as L
from unidiffusion.models import UNet2DConditionModel_UniDiffusion
from unidiffusion.models import StableDiffusionRectifyPipeline
from unidiffusion.config import LazyCall as L
from unidiffusion.datasets import ExcelDataset


dataset = L(ExcelDataset)(
    path=None,
    placeholder=None,
    inversion_placeholder=None,
    excel_path="datasets/body/sku_title.xlsx",
    resolution='${train.resolution}',
    prompt_template="a photo of {}"
)

train.project = "UniDiffusion-Body"
train.gradient_checkpointing = False
train.use_ema = True
optimizer.optimizer = '8bit_adam'
optimizer.lr = 1e-5

train.resolution = (768, 512)
dataset.path = "datasets/body/"
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'


train.checkpointing_iter = 10000
dataset.placeholder = 'body'    # not used in null-text mode
dataset.flip_prob = 0.5

# 2 gpus. Total batch size = 2 * 16 * 1 = 32
dataloader.batch_size = 16
train.gradient_accumulation_iter = 1

# Inference
inference.inference_iter = 2000
inference.rectify_uncond = True
inference.guidance_scale = [1.5 + 1 * i for i in range(5)]
inference.total_num = 20

# Evaluation
evaluation.evaluation_iter = 0
evaluation.rectify_uncond = True
evaluation.evaluator.fid.enabled = True
evaluation.evaluator.fid.real_image_num = 1000
evaluation.total_num = 1000
evaluation.guidance_scale = [1.5 + 0.25 * i for i in range(5)]

train.output_dir = 'experiments/body/body_lora_bs32_1e-5_768-512_fromepic'
train.max_iter = 1000000
train.wandb.enabled = True

inference_pipeline = L(StableDiffusionRectifyPipeline.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
)

unet_init = L(UNet2DConditionModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="epicphotogasm_lastUnicorn",
    subfolder='unet',
)

unet = L(UNet2DConditionModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="epicphotogasm_lastUnicorn",
    subfolder='unet',
    training_args = {
        r'attn2\.(to_k|to_v)': {
            'mode': 'lora',
            'module_kwargs': {'scale': 1.0, 'rank': 4},
            'optim_kwargs': {'lr': '${optimizer.lr}'},
        },
    }
)
