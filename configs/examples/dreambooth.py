from configs.common.get_common_config import *


dataset = get_config("common/data/image_dataset.py").dataset
dataset.path = 'samples/faces'
dataset.inversion_placeholder = '<face>'    # set textual inversion tokens

train.output_dir = 'experiments/faces/lora'
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'


# set mode to 'lora' to enabled dreambooth_lora
unet.training_args = {
    r'attn2\.(to_k|to_v)': {
        'mode': 'lora',
        'module_kwargs': {'scale': 1.0, 'rank': 4},
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}

text_encoder.training_args = {
    'text_embedding': {
        'initial': True,         # whether to init additional token by their text.
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
