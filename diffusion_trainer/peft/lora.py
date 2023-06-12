import torch


class LoRALinear(torch.nn.module):
    pass


class LoRAConv(torch.nn.module):
    pass


class LoRAEmbedding(torch.nn.module):
    pass


def replace_lora(input_module):
    if len(input_module.modules):
        for module in input_module.modules:
            replace_lora(module)
    elif isinstance(input_module, torch.nn.Linear):
        input_module = LoRALinear(input_module)