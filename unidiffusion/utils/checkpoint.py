import diffusers
import transformers
import os
import torch
from unidiffusion.peft.proxy import ProxyNetwork
from unidiffusion.utils.logger import setup_logger


logger = setup_logger(__name__)

UNET_CLASSES = (
    diffusers.UNet2DConditionModel,
)

EMA_CLASSES = (
    diffusers.EMAModel,
)

VAE_CLASSES = (
    diffusers.AutoencoderKL,
)

TEXT_ENCODER_CLASSES = (
    transformers.CLIPTextModel,
)

PROXY_MODEL_CLASS = ProxyNetwork


def get_model_type(model):
    if isinstance(model, UNET_CLASSES):
        return "unet"
    elif isinstance(model, VAE_CLASSES):
        return "vae"
    elif isinstance(model, TEXT_ENCODER_CLASSES):
        return "text_encoder"
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def save_model_hook(models, weights, output_dir):
    for model in models:
        if isinstance(model, PROXY_MODEL_CLASS):
            torch.save(model.state_dict(), os.path.join(output_dir, "proxy_model.pt"))
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        if isinstance(model, PROXY_MODEL_CLASS):
            logger.info("Loading proxy model from %s", input_dir)
            weight = torch.load(os.path.join(input_dir, "proxy_model.pt"), map_location='cpu')
            model.load_state_dict(weight)
