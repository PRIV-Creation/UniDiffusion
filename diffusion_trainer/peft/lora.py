import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LoRALinearLayer(nn.Module):
    proxy_name = 'lora'

    def __init__(self, org_module, rank=4, network_alpha=None, multiplier=1.0):
        super().__init__()

        in_features = org_module.in_features
        out_features = org_module.out_features
        self.org_linear = org_module
        
        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.scale = network_alpha / rank
        self.multiplier = multiplier

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        # self.magic()
        self.org_forward = self.org_linear.forward
    
    def magic(self):
        self.org_forward = self.org_linear.forward
        # self.org_linear.forward = self.forward
        # del self.org_linear

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype) * self.scale * self.multiplier + self.org_forward(hidden_states)


class LoRAConvLayer(nn.Module):
    proxy_name = 'lora_conv'

    def __init__(self, org_module: nn.Module, rank=4, network_alpha=None, multiplier=1.0, dropout=0., use_cp=False):
        super().__init__()
        
        assert isinstance(org_module, nn.Conv2d)

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
        self.org_module = org_module
        self.apply_to()

    def apply_to(self):
        self.org_forward = self.org_module.forward


    def make_weight(self):
        wa = self.lora_up.weight
        wb = self.lora_down.weight
        return (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(self.shape)

    def forward(self, x):
        if self.cp:
            return self.org_forward(x)  + self.dropout(
                self.lora_up(self.lora_mid(self.lora_down(x)))* self.multiplier * self.scale
            )
        else:
            return self.org_forward(x)  + self.dropout(
                self.lora_up(self.lora_down(x))* self.multiplier * self.scale
            )