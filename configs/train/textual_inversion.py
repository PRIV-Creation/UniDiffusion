from configs.common.get_common_config import *


dataset = get_config("common/data/image_dataset.py").dataset
dataset.path = 'samples/faces'
dataset.inversion_placeholder = '<face>'    # set textual inversion tokens

train.output_dir = 'experiments/faces/lora'
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

text_encoder.training_args = {
    'text_embedding': {
        'initial': True,         # whether to init additional token by their text.
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
