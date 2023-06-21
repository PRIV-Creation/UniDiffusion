from .base_model import BaseModel
from diffusers import AutoencoderKL


class AutoencoderKL_DT(BaseModel, AutoencoderKL):
    model_name = 'vae'

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

