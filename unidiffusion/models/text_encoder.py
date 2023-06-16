from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_DT(BaseModel, CLIPTextModel):
    model_name = 'text_encoder'

    def check_validate(self):
        pass
