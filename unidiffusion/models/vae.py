from .base_model import BaseModel
from diffusers import AutoencoderKL


class AutoencoderKL_DT(BaseModel, AutoencoderKL):
    model_name = 'vae'

