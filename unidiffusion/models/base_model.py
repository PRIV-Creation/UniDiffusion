import re
import torch
import itertools
from abc import ABC
from omegaconf import OmegaConf
from unidiffusion.peft.lora import set_lora_layer
from unidiffusion.peft.finetune import set_finetune_layer
from unidiffusion.utils.module_regular_search import get_module_pattern
from unidiffusion.utils.logger import setup_logger


class BaseModel:
    trainable = False
    params_train_args = dict()
    params_group = []
    proxy_name = None
    model_name: str

    @classmethod
    def from_pretrained(cls, proxy_model=None, training_args=None, *args, **kwargs):
        cls.trainable = training_args is not None
        cls.params_train_args = training_args
        setup_logger(__name__).info('Model {} trainable: {}.'.format(cls.__name__, cls.trainable))
        model = super().from_pretrained(*args, **kwargs)
        if cls.trainable:
            model.parse_training_args(proxy_model)
        return model

    def get_trainable_params(self):
        if self.trainable:
            return self.params_group
        else: 
            return None

    def parse_training_args(self, proxy_model):
        self.requires_grad_(False)
        for pattern, train_args in self.params_train_args.items():
            # params = []
            for module, name in get_module_pattern(self, pattern):
                if (mode := train_args['mode']) == 'finetune':
                    set_finetune_layer(self.model_name, module, name, train_args, proxy_model)
                elif mode == 'lora':
                    set_lora_layer(self.model_name, module, name, train_args, proxy_model)
