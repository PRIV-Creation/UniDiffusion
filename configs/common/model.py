from diffusers import DDPMScheduler, UNet2DConditionModel
from unidiffusion.config import LazyCall as L
from unidiffusion.models import (
    CLIPTextModel_UniDiffusion,
    CLIPTokenizer_UniDiffusion,
    UNet2DConditionModel_UniDiffusion,
    AutoencoderKL_UniDiffusion,
)

# set model
vae = L(AutoencoderKL_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="vae",
)

unet = L(UNet2DConditionModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="unet",
)

unet_init = L(UNet2DConditionModel.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="unet",
)

tokenizer = L(CLIPTokenizer_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="tokenizer",
)

text_encoder = L(CLIPTextModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="text_encoder",
)

noise_scheduler = L(DDPMScheduler.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="scheduler",
)
