from configs.common.get_common_config import *

# update configs
optimizer.optimizer = "AdamW"
optimizer.lr = 1e-4

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

unet.training_args = {
    # update all cross attention by lora
    r'attn2': {
        'mode': 'lora',
        'module_kwargs': {'scale': 1.0, 'rank': 4},
        'optim_kwargs': {'lr': '${optimizer.lr}'},
    },
}