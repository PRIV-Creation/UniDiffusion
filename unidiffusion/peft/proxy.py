import torch
from unidiffusion.utils.module_regular_search import get_model_by_relative_name
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)


class BaseProxy(torch.nn.Module):
    proxy_name: str


class ProxyLayer(torch.nn.Module):
    CAN_BE_MERGED = True
    original_name: str

    # def __init__(self, original_module, proxy_module):
    #     super().__init__()
    #     self.original_module = original_module
    #     self.proxy_module = proxy_module

    def merge_layer(self):
        pass


class ProxyNetwork(torch.nn.Module):
    params_group = []

    def __init__(self):
        super().__init__()
        self.unet = torch.nn.ModuleList()
        self.vae = torch.nn.ModuleList()
        self.text_encoder = torch.nn.ModuleList()

    def set_requires_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def merge_model(self, original_model, model_name):
        proxy_model = getattr(self, model_name)
        for proxy_layer in proxy_model:
            original_layer = get_model_by_relative_name(original_model, proxy_layer.original_name)
            status = proxy_layer.merge_layer(original_layer)
            if not status:
                logger.warn(f"Failed to merge {proxy_layer.original_name}.")
