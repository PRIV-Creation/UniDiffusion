import tokenizers.trainers

from configs.common.get_common_config import *


# Demonstrate to how to update configs
#
optimizer.lr = 1e-4
dataloader.batch_size = 2
train.use_xformers = True

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

tokenizers.training_args = {'placeholders': None}   # placeholders will be set from datasets.

text_encoder.training_args = {
    'text_embedding': {
        'placeholders': None,    # placeholders will be set from datasets.
        'optim_kwargs': {'lr': optimizer.lr}
    }
}
