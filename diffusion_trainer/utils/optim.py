import torch
import torch
import copy
from typing import Any, Dict, List, Set, Optional, Callable


class Optimizer(torch.optim.optimizer):
    def __init__(self, optimizer="", **kwargs):
        optimizer_map = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
        }
        self.optimizer = optimizer_map[optimizer.lower()](**kwargs)

    def __setstate__(self, state: dict):
        self.optimizer.__setstate__(state)

    def state_dict(self) -> dict:
        self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool):
        self.optimizer.zero_grad(set_to_none)

    def step(self, closure):
        self.optimizer.step(closure)

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
    num_layers = None,
    lr_decay_rate = 1.,
    weight_decay_embed = 0.,
    backbone_lr_factor = 1.,
):
    assert lr_decay_rate <= 1.0

    defaults = {}
    defaults["lr"] = base_lr
    defaults["weight_decay"] = weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * backbone_lr_factor
            if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                print(module_param_name)
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed
            params.append({"params": [value], **hyperparams})

    return params