import torch


def get_optimizer(optimizer='SGD', **kwargs):
    if optimizer == '8bit_adam':
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_map = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
        }
        optimizer_class = optimizer_map[optimizer.lower()]
    return optimizer_class(**kwargs)
