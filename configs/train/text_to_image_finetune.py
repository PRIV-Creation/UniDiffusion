from configs.common.get_common_config import *


# Demonstrate to how to update configs
optimizer.lr = 1e-4
dataloader.batch_size = 2

# not use textual inversion
train.use_xformers = True

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

unet.training_args = {
    '': {
        'mode': 'finetune',
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}