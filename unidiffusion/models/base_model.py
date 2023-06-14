import re
import torch
from unidiffusion.utils.logger import setup_logger


class BaseModel:
    trainable = False
    params_train_args = dict()
    params_group = []

    @classmethod
    def from_pretrained(cls, training_args=None, *args, **kwargs):
        cls.trainable = training_args is not None
        cls.params_train_args = training_args
        setup_logger(__name__).info('Model {} trainable: {}.'.format(cls.__name__, cls.trainable))
        model = super().from_pretrained(*args, **kwargs)
        model.parse_training_args()
        return model

    def parse_training_args(self):
        pass

    def get_trainable_params(self):
        if self.trainable:
            return self.params_group
        else: 
            return None

    def set_proxy_layer(self, **kwargs):
        return NotImplementedError()

    def get_all_proxy_layers(self):
        return NotImplementedError()


def get_trainable_module(input_module, pattern):
    last_match_name = "LAST MATCHED MODULE'S NAME"
    for name, module in input_module.named_modules():
        if bool(re.search(pattern, name)):
            if name.startswith(last_match_name):
                continue
            last_match_name = name
            yield input_module, name
