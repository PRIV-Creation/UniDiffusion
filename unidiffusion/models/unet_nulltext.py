from .base_model import BaseModel
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from unidiffusion.peft.null_text import NullTextAttention
import torch
from unidiffusion.utils.logger import setup_logger


class UNet2DConditionModel_NullText(BaseModel, UNet2DConditionModel):
    model_name = 'unet'

    @classmethod
    def from_pretrained(cls, proxy_model=None, *args, **kwargs):
        training_args = kwargs.pop("training_args", None)
        unet = super().from_pretrained(*args, **kwargs)
        unet.trainable = training_args is not None
        unet.params_train_args = training_args
        for name, module in unet.named_modules():
            if name.endswith("attn2"):
                _ = NullTextAttention(module, initial=True)
                channel = module.to_q.in_features
                del module.to_q
                del module.to_k
                del module.to_v
                del module.to_out
                module.register_buffer("null_text_feature", torch.zeros([1, 1, channel]))
            if isinstance(module, BasicTransformerBlock):
                module.norm2 = torch.nn.Identity()
        unet.load_state_dict(torch.load(kwargs["null_text_checkpoint"], map_location=unet.device), strict=False)
        if unet.trainable:
            unet.parse_training_args(proxy_model)
        setup_logger(__name__).info('Model {} trainable: {}.'.format(cls.model_name, cls.trainable))
        return unet