from configs.common.get_common_config import *


# Demonstrate to how to update configs
#
optimizer.lr = 1e-4
dataloader.batch_size = 2
train.use_xformers = True

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

text_encoder.training_args = {
    'text_embedding': {
        'initial': True,         # whether to init additional token by their text.
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
