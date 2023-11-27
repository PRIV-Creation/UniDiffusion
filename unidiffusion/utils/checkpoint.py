import diffusers
import transformers
import os
import glob
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
            model_state_dict = model.state_dict()
            # save text embedding
            if (embedding := model_state_dict.pop('text_embeddings.weight', None)) is not None:
                os.makedirs(os.path.join(output_dir, "text_embedding"), exist_ok=True)
                for placeholder, index in model.added_tokens_encoder.items():
                    placepolder_inversion_embedding = embedding[index]
                    torch.save(placepolder_inversion_embedding, os.path.join(output_dir, f"text_embedding/{placeholder}.pt"))
            # save proxy model
            if len(model_state_dict) > 0:
                torch.save(model_state_dict, os.path.join(output_dir, "proxy_model.pt"))
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        if isinstance(model, PROXY_MODEL_CLASS):
            logger.info("Loading proxy model from %s", input_dir)
            # Load text embedding
            if os.path.exists(os.path.join(input_dir, "text_embedding")):
                for embedding_path in glob.glob(os.path.join(input_dir, "text_embedding/*.pt")):
                    embedding = torch.load(embedding_path).to(model.text_embeddings.weight.device)
                    placeholder = os.path.basename(embedding_path).replace(".pt", "")
                    index = model.added_tokens_encoder[placeholder]
                    model.text_embeddings.weight.data[index] = embedding
                    logger.info(f"Loading text embedding: {placeholder}")
                    del embedding
            # Load proxy model
            if os.path.exists(os.path.join(input_dir, "proxy_model.pt")):
                weight = torch.load(os.path.join(input_dir, "proxy_model.pt"), map_location='cpu')
                incompatible_keys = model.load_state_dict(weight, strict=False)
                missing_keys = [k for k in incompatible_keys.missing_keys if "text_embeddings.weight" not in k]
                if len(missing_keys) > 0:
                    logger.warning("Missing keys when loading proxy model: %s", ','.join(missing_keys))
                if len(unexpected_keys := incompatible_keys.unexpected_keys) > 0:
                    logger.warning("Unexpected keys when loading proxy model: %s", ','.join(unexpected_keys))
                del weight
        torch.cuda.empty_cache()
