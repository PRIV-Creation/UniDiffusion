from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_DT(BaseModel, CLIPTextModel):
    def from_pretrained(self, training_args=None, *args, **kwargs):
        self.training = training_args is None or len(training_args) > 0
        self.params_args = training_args
        return super().from_pretrained(*args, **kwargs)