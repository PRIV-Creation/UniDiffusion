from .base_model import BaseModel
from diffusers import UNet2DConditionModel


class UNet2DConditionModel_DT(BaseModel, UNet2DConditionModel):
    model_name = 'unet'
