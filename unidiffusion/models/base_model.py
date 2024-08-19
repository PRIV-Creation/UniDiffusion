from unidiffusion.peft import set_lora_layer, set_finetune_layer
from unidiffusion.utils.module_regular_search import get_module_pattern
from unidiffusion.utils.logger import setup_logger


class BaseModel:
    trainable = False
    params_train_args = dict()
    params_group = []
    model_name: str

    @classmethod
    def from_pretrained(cls, proxy_model=None, training_args=None, *args, **kwargs):
        #################################################################################
        ### model = super().from_pretrained(*args, **kwargs) 
        ### 调用unet2dcondition里继承的ModelMixin的装饰器包过的from_pretrained,太复杂没懂
        ### 但返回的是UNet2DConditionModel_UniDiffusion即models里面的类
        #################################################################################
        model = super().from_pretrained(*args, **kwargs) 
        model.trainable = training_args is not None
        model.params_train_args = training_args
        if model.trainable:
            model.parse_training_args(proxy_model)
        setup_logger(__name__).info('Model {} trainable: {}.'.format(model.model_name, model.trainable))
        return model

    def get_trainable_params(self):
        if self.trainable:
            return self.params_group
        else: 
            return None

    def parse_training_args(self, proxy_model):
        self.requires_grad_(False)
        for pattern, train_args in self.params_train_args.items():
            for module, name in get_module_pattern(self, pattern):
                if (mode := train_args['mode']) == 'finetune':
                    set_finetune_layer(self.model_name, module, name, train_args, proxy_model)
                elif mode == 'lora':
                    set_lora_layer(self.model_name, module, name, train_args, proxy_model)
