from configs.common.get_common_config import *


dataset = get_config("common/data/huggingface_dataset.py").dataset

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

unet.training_args = {
    '': {
        'mode': 'finetune',
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}