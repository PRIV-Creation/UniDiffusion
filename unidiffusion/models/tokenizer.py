from .base_model import BaseModel
from transformers import CLIPTokenizer


class CLIPTokenizer_DT(CLIPTokenizer, BaseModel):

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def set_placeholders(self, placeholders):
        num_added_tokens = self.add_tokens(placeholders)
