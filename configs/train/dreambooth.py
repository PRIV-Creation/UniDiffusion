from configs.get_common_config import *


# update configs
optimizer.optimizer = "SGD"
optimizer.lr = 2e-7

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

unet.training_args = {'': {'mode': 'finetune', 'optim_kwargs': {'lr': optimizer.lr}}}