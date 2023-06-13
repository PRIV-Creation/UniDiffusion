from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_DT(BaseModel, CLIPTextModel):
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    
    def check_validate(self):
        pass

    