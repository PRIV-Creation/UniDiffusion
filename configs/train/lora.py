from configs.get_common_config import *


# update configs
optimizer.optimizer = "AdamW"
optimizer.lr = 2e-7

train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
train.use_xformers = False
# unet.training_args = {'Transformer2DModel.Linear': 
#                       {'mode': 'lora', 'extra_args': {'network_alpha': 0.1, 'rank': 4, 'bias': False}},
#                       'Attention.Linear': 
#                       {'mode', 'lora'}
#                       }
unet.training_args = {  'Attention.Linear': 
                        {
                            'mode': 'lora', 
                            'extra_args': {'network_alpha': 0.1, 'rank': 4}
                        },
                        'ResnetBlock2D.Conv2d':
                        {
                            'mode': 'lora',
                            'extra_args': {'network_alpha': 0.1, 'rank': 4}
                        }       
                    }
