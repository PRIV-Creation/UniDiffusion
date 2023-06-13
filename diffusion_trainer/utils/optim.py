import torch


def get_optimizer(optimizer='SGD', **kwargs):
    if optimizer == '8bit_adam':
        import bitsandbytes as bnb

    optimizer_map = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        '8bit_adam': bnb.optim.AdamW8bit
    }
    return optimizer_map[optimizer.lower()](**kwargs)
