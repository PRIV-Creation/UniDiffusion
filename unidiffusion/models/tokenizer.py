from .base_model import BaseModel
from transformers import CLIPTokenizer


class CLIPTokenizer_UniDiffusion(CLIPTokenizer, BaseModel):

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def set_placeholders(self, placeholders):
        for p in placeholders:
            self.add_tokens(p)
