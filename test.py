from diffusers import UNet2DConditionModel
from diffusers import ConfigMixin


unet1 = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet')
print()