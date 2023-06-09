import torch


class BaseModel(torch.nn.Module):
    training = False,
    params_args = dict()
