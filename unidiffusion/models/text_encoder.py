from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_DT(BaseModel, CLIPTextModel):
    model_name = 'text_encoder'

    def parse_training_args(self, proxy_model):
        pass
