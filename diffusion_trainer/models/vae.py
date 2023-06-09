from .base_model import BaseModel
from diffusers import AutoencoderKL


class AutoencoderKL_DT(BaseModel, AutoencoderKL):
    def from_pretrained(self, training_args=None, *args, **kwargs):
        self.training = training_args is None or len(training_args) > 0
        self.params_args = training_args
        return super().from_pretrained(*args, **kwargs)