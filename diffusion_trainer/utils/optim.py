import torch
import torch
import copy
from typing import Any, Dict, List, Set, Optional, Callable


def get_optimizer(optimizer='SGD', **kwargs):
    optimizer_map = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }
    return optimizer_map[optimizer.lower()](**kwargs)
