import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from unidiffusion.utils.module_regular_search import get_module_type
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)


LORA_SUPPORTED_MODULES = (
    nn.Linear,
    nn.Conv2d,
)


class BaseLoRAModule(nn.Module):
    proxy_name: str
    target_replace_layer: str

    def __init__(self, org_module: nn.Module, **kwargs) -> None:
        super().__init__()
        self.org_module = org_module
    
    def apply_to(self):
        self.org_forward = self.org_module.forward

    def set_trainable(self):
        self.requires_grad_(True)
        self.org_module.requires_grad_(False)

    def get_trainable_parameters(self):
        for name, params in self.named_parameters():
            if params.requires_grad:
                yield params
    
    def named_trainable_parameters(self):
        for name, params in self.named_parameters():
            if params.requires_grad:
                yield name, params


class LoRALinearLayer(BaseLoRAModule):
    proxy_name = 'lora'
    target_replace_layer = 'Linear'

    def __init__(self, org_module, rank=4, scale=1.0):
        super().__init__(org_module)

        in_features = org_module.in_features
        out_features = org_module.out_features
        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.weight = org_module.weight
        self.bias = org_module.bias

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.rank = rank
        self.scale = scale

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        # control what params to be grad-enabled
        self.set_trainable()
        self.apply_to()

    # @property
    # def trainable_parameters(self):
    #     return itertools.chain(self.down.parameters(), self.up.parameters())

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype) * self.scale + F.linear(hidden_states, self.weight, self.bias)


class LoRAConvLayer(BaseLoRAModule):
    proxy_name = 'lora_conv'
    target_replace_layer = 'Conv2d'

    def __init__(self, org_module: nn.Module, rank=4, network_alpha=None, multiplier=1.0, dropout=0., use_cp=False):
        assert isinstance(org_module, nn.Conv2d)
        super().__init__(org_module)
        in_dim = org_module.in_channels
        k_size = org_module.kernel_size
        stride = org_module.stride
        padding = org_module.padding
        out_dim = org_module.out_channels
        if use_cp and k_size != (1, 1):
            self.lora_down = nn.Conv2d(in_dim, rank, (1, 1), bias=False)
            self.lora_mid = nn.Conv2d(rank, rank, k_size, stride, padding, bias=False)
            self.cp = True
        else:
            self.lora_down = nn.Conv2d(in_dim, rank, k_size, stride, padding, bias=False)
            self.cp = False
        self.lora_up = nn.Conv2d(rank, out_dim, (1, 1), bias=False)
        self.shape = org_module.weight.shape
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        if type(network_alpha) == torch.Tensor:
            network_alpha = network_alpha.detach().float().numpy()  # without casting, bf16 causes error
        network_alpha = rank if network_alpha is None or network_alpha == 0 else network_alpha
        self.scale = network_alpha / rank
        self.register_buffer('alpha', torch.tensor(network_alpha))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        if self.cp:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))
        self.multiplier = multiplier

        self.set_trainable()
        self.apply_to()

    def make_weight(self):
        wa = self.lora_up.weight
        wb = self.lora_down.weight
        return (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(self.shape)

    def forward(self, x):
        if self.cp:
            return self.org_forward(x) + self.dropout(
                self.lora_up(self.lora_mid(self.lora_down(x)))* self.multiplier * self.scale
            )
        else:
            return self.org_forward(x) + self.dropout(
                self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
            )


def lora_proxy(module, lora_args):
    if isinstance(module, nn.Linear):
        m = LoRALinearLayer(module, **lora_args)
        return m.get_trainable_parameters(), m
    elif isinstance(module, nn.Conv2d):
        m = LoRAConvLayer(module, **lora_args)
        return m.get_trainable_parameters(), m
    else:
        raise ValueError(f"LoRA does not support {type(module)}")


def get_lora_proxy_layer(input_module, lora_args, input_name):
    trainable_params = None
    for module, name in get_module_type(input_module, LORA_SUPPORTED_MODULES):
        names = name.split('.')
        if names[0] == '':
            logger.debug(f'LoRA proxy layer: {input_name}')
            return lora_proxy(input_module, lora_args)
        layer_instance = input_module
        for i, layer_name in enumerate(names):
            if i < len(names) - 1:
                if layer_name.isdigit():
                    layer_instance = layer_instance[int(layer_name)]
                else:
                    layer_instance = getattr(layer_instance, layer_name)
            else:
                lora_proxy_name = f'{input_name}.{name}' if name != '' else input_name
                logger.debug(f'LoRA proxy layer: {lora_proxy_name}')
                trainable_params, proxy_layer = lora_proxy(module, lora_args)
                setattr(layer_instance, layer_name, proxy_layer)
    return trainable_params, input_module
