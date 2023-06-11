from .base_model import BaseModel
from transformers import CLIPTokenizer


class CLIPTokenizer_DT(CLIPTokenizer, BaseModel):
    pass
