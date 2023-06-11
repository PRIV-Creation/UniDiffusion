from .base_model import BaseModel
from diffusers import AutoencoderKL


class AutoencoderKL_DT(BaseModel, AutoencoderKL):
    @classmethod
    def from_pretrained(cls, training_args=None, *args, **kwargs):
        cls.training = training_args is None or len(training_args) > 0
        cls.params_args = training_args
        return super().from_pretrained(*args, **kwargs)