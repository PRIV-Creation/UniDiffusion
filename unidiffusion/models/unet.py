import itertools
from .base_model import BaseModel
from omegaconf import OmegaConf
from diffusers import UNet2DConditionModel
from unidiffusion.peft.lora import get_lora_proxy_layer
from unidiffusion.utils.module_regular_search import get_module_pattern
import torch.nn as nn


class UNet2DConditionModel_DT(BaseModel, UNet2DConditionModel):
    def parse_training_args(self):
        self.requires_grad_(False)
        for pattern, train_args in self.params_train_args.items():
            params = []
            for module, name in get_module_pattern(self, pattern):
                trainable_params = self.set_proxy_layer(module, name, train_args)
                params.append(trainable_params)
            if 'optim_kwargs' in train_args:
                optim_params = OmegaConf.to_container(train_args['optim_kwargs'])
            else:
                optim_params = dict()
            optim_params['params'] = itertools.chain(*params)
            self.params_group.append(optim_params)

    def set_proxy_layer(self, module, name, train_args):
        # get proxy layer
        mode = train_args['mode']
        proxy_layer_kwargs = train_args['layer_kwargs']
        if mode == 'finetune':
            trainable_params, proxy_layer = module.parameters(), module
        elif mode == 'lora':
            trainable_params, proxy_layer = get_lora_proxy_layer(module, proxy_layer_kwargs, name)
        else:
            raise NotImplementedError(f'Unknown mode: {mode}')

        # parse name
        names = name.split('.')
        layer_instance = self
        for i, layer_name in enumerate(names):
            if i < len(names) - 1:
                if layer_name.isdigit():
                    layer_instance = layer_instance[int(layer_name)]
                else:
                    layer_instance = getattr(layer_instance, layer_name)
            else:
                # TODO
                setattr(layer_instance, layer_name, proxy_layer)
        return trainable_params
