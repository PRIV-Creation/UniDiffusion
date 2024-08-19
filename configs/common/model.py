from diffusers import DDPMScheduler, UNet2DConditionModel
from unidiffusion.config import LazyCall as L
from unidiffusion.models import (
    CLIPTextModel_UniDiffusion,
    CLIPTokenizer_UniDiffusion,
    UNet2DConditionModel_UniDiffusion,
    AutoencoderKL_UniDiffusion,
    CLIPTextModelWithProjection_UniDiffusion
)

# set model #没干别的 就是设置了个字典而已
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

tokenizer_one = L(CLIPTokenizer_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="tokenizer",
)

tokenizer_two = L(CLIPTokenizer_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="tokenizer_2",
)

text_encoder_one = L(CLIPTextModel_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="text_encoder",
)

text_encoder_two = L(CLIPTextModelWithProjection_UniDiffusion.from_pretrained)(
    pretrained_model_name_or_path="${..train.pretrained_model_name_or_path}",
    subfolder="text_encoder_2",
)