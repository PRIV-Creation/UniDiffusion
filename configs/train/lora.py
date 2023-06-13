from configs.get_common_config import *


# update configs
optimizer.optimizer = "AdamW"
optimizer.lr = 2e-7

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
train.use_xformers = False

unet.training_args = {  'Attention.Linear': 
                        {
                            'mode': 'lora', 
                            'layer_kwargs': {'network_alpha': 0.1, 'rank': 4},
                            'lr': 1e-4
                        },

                        'ResnetBlock2D.Conv2d':
                        {
                            'mode': 'lora',
                            'layer_kwargs': {'network_alpha': 0.1, 'rank': 4},
                            'lr': 1e-3
                        }
                    }

# text_encoder.training_args = {'*': {'mode': 'finetune', 'lr': 1e-4}}