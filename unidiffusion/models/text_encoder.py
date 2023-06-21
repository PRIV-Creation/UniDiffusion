import torch
from .base_model import BaseModel
from transformers import CLIPTextModel


class CLIPTextModel_DT(BaseModel, CLIPTextModel):
    model_name = 'text_encoder'

    def set_placeholders(self, placeholders, tokenizer, proxy_model):
        if self.params_train_args is None:
            return None
        # add text embedding
        if (text_embedding_args := self.params_train_args.pop('text_embedding', None)) is not None:
            if text_embedding_args.get('initial', True):
                ori_token_embeds = self.get_input_embeddings().weight.data
                additional_embedding = []
                start_idx = tokenizer.convert_tokens_to_ids(placeholders[0])
                for p in placeholders:
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

        # set embedding to trainable
        self.text_model.embeddings.token_embedding.requires_grad_(True)
        proxy_model.text_embeddings = self.text_model.embeddings.token_embedding
        proxy_model.params_group.append(dict(params=proxy_model.text_embeddings.weight, **text_embedding_args['optim_kwargs']))
