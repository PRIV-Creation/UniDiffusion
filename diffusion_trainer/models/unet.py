from .base_model import BaseModel
from diffusers import UNet2DConditionModel


class UNet2DConditionModel_DT(BaseModel, UNet2DConditionModel):
    def from_pretrained(self, training_args=None, *args, **kwargs):
        self.training = training_args is None or len(training_args) > 0
        self.params_args = training_args
        return super().from_pretrained(*args, **kwargs)

    def get_params(self):
        return self.parameters()