import itertools
from .base_model import BaseModel
from omegaconf import OmegaConf
from diffusers import UNet2DConditionModel

import torch.nn as nn


class UNet2DConditionModel_DT(BaseModel, UNet2DConditionModel):
    model_name = 'unet'
