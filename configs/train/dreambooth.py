from diffusion_trainer.config import get_config
from diffusion_trainer.utils.optim import Optimizer
from diffusion_trainer.config import LazyCall as L
from diffusion_trainer.utils.optim import get_default_optimizer_params
from diffusion_trainer.models import (
    CLIPTextModel_DT,
    CLIPTokenizer_DT,
    UNet2DConditionModel_DT,
    AutoencoderKL_DT,
)

# get default config
dataloader = get_config("common/data/txt_dataset.py").optimizer
optimizer = get_config("common/optim.py").optimizer
train = get_config("common/train.py").train
accelerator = get_config("common/train.py").accelerator
vae = get_config("common/model.py").vae
unet = get_config("common/model.py").unet
tokenizer = get_config("common/model.py").tokenizer
text_model = get_config("common/model.py").text_model

# update configs
optimizer.optimizer = "AdamW"
optimizer.lr = 2e-7

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

