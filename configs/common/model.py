from diffusers import DDPMScheduler
from diffusion_trainer.config import LazyCall as L
from diffusion_trainer.models import (
    CLIPTextModel_DT,
    CLIPTokenizer_DT,
    UNet2DConditionModel_DT,
    AutoencoderKL_DT,
)

# set model
vae = L(AutoencoderKL_DT)(
    _function_="from_pretrained",
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="vae",
)

unet = L(UNet2DConditionModel_DT)(
    _function_="from_pretrained",
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="unet",
)

tokenizer = L(CLIPTokenizer_DT)(
    _function_="from_pretrained",
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="tokenizer",
)

text_encoder = L(CLIPTextModel_DT)(
    _function_="from_pretrained",
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="text_encoder",
)

noise_scheduler = L(DDPMScheduler)(
    _function_="from_pretrained",
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="scheduler",
)