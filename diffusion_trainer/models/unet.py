from typing import Union
from .base_model import BaseModel
from diffusers import UNet2DConditionModel
from ..peft.lora import LoRALinearLayer, LoRAConvLayer
import torch.nn as nn

class UNet2DConditionModel_DT(BaseModel, UNet2DConditionModel):

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]

    def parse_training_args(self):
        # "*.Linear": {"mode": "lora", "network_alpha": 0.1, "rank": 4, "bias": False}
        self.target_modules = [k.split('.')[0] for k in self.params_train_args.keys()]
        self.target_replace_layers = [k.split('.')[1] for k in self.params_train_args.keys()]
        self.methods = [v for v in self.params_train_args.values()]
        self.named_proxy_modules = {}
        self.trainable_params = []


    def default_scope(self):
        return ['Attention']


    def set_proxy_layer(self, 
                        layer_name,
                        origin_layer: nn.Module,
                        target_replace_layer: str,
                        mode: str,
                        proxy_layer_kwargs: dict = None,
                        ):
        nn_module_name = origin_layer.__class__.__name__
        if mode == 'finetune':
            return origin_layer
        elif mode == 'lora':
            if nn_module_name != target_replace_layer:
                return
            if nn_module_name == 'Linear':
                proxy_layer = LoRALinearLayer(origin_layer, **proxy_layer_kwargs)
            elif nn_module_name == 'Conv2d':
                proxy_layer = LoRAConvLayer(origin_layer, **proxy_layer_kwargs)
            proxy_layer.requires_grad_(True)
            self.trainable_params.extend([v for _, v in proxy_layer.named_parameters()])
            return proxy_layer
        
        
    def get_all_proxy_layers(self):
        return self.named_proxy_modules


    def get_trainable_params(self):
        self.requires_grad_(False)
        self.parse_training_args()
        for name, module in self.named_modules():
            # for target_module, target_replace_layer, method in zip(self.target_modules, self.target_replace_layers, self.methods):
            module_name = module.__class__.__name__
            if module_name in self.target_modules:
                target_replace_layer = self.target_replace_layers[self.target_modules.index(module_name)]
                method = self.methods[self.target_modules.index(module_name)]
                mode = method['mode']
                proxy_layer_kwargs = method['extra_args']
                for layer_name, layer in module.named_modules():
                    proxy_layer = self.set_proxy_layer(layer_name, layer, target_replace_layer, mode, proxy_layer_kwargs)
                    if proxy_layer is not None:
                        module.__setattr__(layer_name, proxy_layer)
                        self.named_proxy_modules.update({f'{name}.{layer_name}.{proxy_layer.proxy_name}': proxy_layer})
        for name, params in self.named_parameters():
            if params.requires_grad:
                print(name)
        for name, proxy_layer in self.named_proxy_modules.items():
            print(name)
        return params