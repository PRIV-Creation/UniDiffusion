from configs.get_common_config import *

# update configs
optimizer.optimizer = "AdamW"
optimizer.lr = 2e-7

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
train.use_xformers = False

unet.training_args = {
    r'attn1': {
        'mode': 'lora',
        'layer_kwargs': {'scale': 1.0, 'rank': 4},
        'optim_kwargs': {'lr': optimizer.lr},
    },
}