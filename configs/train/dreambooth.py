from configs.common.get_common_config import *
from unidiffusion.datasets.dreambooth_dataset import collate_fn

dataset = get_config("common/data/image_dataset.py").dataset
dataset.path = 'samples/faces'
dataset.inversion_placeholder = '<face>'    # set textual inversion tokens

# db dataset
dataset = get_config("common/data/dreambooth_dataset.py").dataset
dataset.instance_data_root = 'samples/faces'
dataset.instance_prompt = 'A photo of a <face>.'


dataloader.collate_fn = collate_fn

train.output_dir = 'experiments/faces/lora'
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
# use dreambooth regularization
train.db.with_prior_preservation = True
train.db.class_data_dir = 'prior_data/faces'
train.db.class_prompt = 'face.'
train.db.num_class_images = 100

dataset.class_data_root = train.db.class_data_dir
dataset.class_num = train.db.num_class_images
dataset.class_prompt = train.db.class_prompt

# set mode to 'lora' to enabled dreambooth_lora
unet.training_args = {
    '': {
        'mode': 'finetune',
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}

text_encoder.training_args = {
    'text_embedding': {
        'initial': True,         # whether to init additional token by their text.
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
