from .base_model import BaseModel
from diffusers import UNet2DConditionModel


class UNet2DConditionModel_UniDiffusion(BaseModel, UNet2DConditionModel):
    model_name = 'unet'

    @classmethod
    def from_pretrained(cls, proxy_model=None, *args, **kwargs):
        unet = super().from_pretrained(proxy_model=proxy_model, *args, **kwargs)
        return unet