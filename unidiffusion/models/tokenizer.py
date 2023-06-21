from .base_model import BaseModel
from transformers import CLIPTokenizer


class CLIPTokenizer_DT(CLIPTokenizer, BaseModel):

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def set_placeholders(self, placeholders):
        num_added_tokens = self.add_tokens(placeholders)
        # if num_added_tokens != len(placeholders):
        #     raise ValueError(
        #         f"The tokenizer already contains the token. Please pass a different"
        #         " `placeholder_token` that is not already in the tokenizer."
        #     )
