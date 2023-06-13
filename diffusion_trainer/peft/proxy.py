import torch


class BaseProxy(torch.nn.Module):
    proxy_name: str