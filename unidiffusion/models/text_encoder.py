import torch
from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_UniDiffusion(BaseModel, CLIPTextModel):
    model_name = 'text_encoder'
    start_token_idx = None
    end_token_idx = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def set_placeholders(self, placeholders, tokenizer, proxy_model):
        # add text embedding
        if self.params_train_args is not None and (text_embedding_args := self.params_train_args.pop('text_embedding', None)) is not None:
            if text_embedding_args.get('initial', True):
                ori_token_embeds = self.get_input_embeddings().weight.data
                additional_embedding = []
                start_idx = tokenizer.convert_tokens_to_ids(placeholders[0])
                for p in placeholders:
                    if p in (additional_args := text_embedding_args.get('additional_args', dict())):
                        initial_text = additional_args[p].get('initial', False)
                        if initial_text:
                            token_ids = tokenizer.encode(initial_text, add_special_tokens=False)
                        else:
                            continue
                    else:
                        if p[0] == '<' and p[-1] == '>':
                            p = p[1: -1]
                        token_ids = tokenizer.encode(p, add_special_tokens=False)
                    additional_embedding.append(ori_token_embeds[token_ids].mean(dim=0))
            else:
                start_idx, additional_embedding = None, None

            self.resize_token_embeddings(len(tokenizer))
            if text_embedding_args.get('initial', True):
                token_embeds = self.get_input_embeddings().weight.data
                token_embeds[start_idx:] = torch.stack(additional_embedding)
            self.start_token_idx = start_idx
            self.end_token_idx = start_idx
            train_embedding = True
        elif placeholders is not None:
            # use inverse placeholders but not trainable
            self.resize_token_embeddings(len(tokenizer))
            self.start_token_idx = None
            self.end_token_idx = None
            train_embedding = False
        else:
            train_embedding = False

        # mapping dictionary
        proxy_model.added_tokens_encoder = tokenizer.added_tokens_encoder.copy()
        proxy_model.added_tokens_decoder = tokenizer.added_tokens_decoder.copy()

        # set embedding to trainable
        proxy_model.text_embeddings = self.text_model.embeddings.token_embedding
        if train_embedding:
            self.text_model.embeddings.token_embedding.requires_grad_(True)
            proxy_model.params_group.append(dict(params=proxy_model.text_embeddings.weight, **text_embedding_args['optim_kwargs']))
