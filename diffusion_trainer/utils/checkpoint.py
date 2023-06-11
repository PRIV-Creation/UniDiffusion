import diffusers
import transformers
import os


UNET_CLASSES = [
    diffusers.UNet2DModel,
]

EMA_CLASSES = [
    diffusers.EMAModel,
]

VAE_CLASSES = [
    diffusers.AutoencoderKL,
]

TEXT_ENCODER_CLASSES = [
    transformers.CLIPTextModel,
]


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
        model_type = get_model_type(model)
        is_ema = isinstance(model, EMA_CLASSES)
        sub_dir = model_type if not is_ema else f"{model_type}_ema"
        model.save_pretrained(os.path.join(output_dir, sub_dir))
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        model_type = get_model_type(model)
        is_ema = isinstance(model, EMA_CLASSES)
        model_name = model_type if not is_ema else f"{model_type}_ema"

        load_model = type(model).from_pretrained(input_dir, subfolder=model_name)
        if model_type == 'unet':
            model.register_to_config(**load_model.config)
        else:
            model.config = load_model.config

        model.load_state_dict(load_model.state_dict())
        del load_model
