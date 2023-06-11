from diffusion_trainer.config import get_config


# get default config
dataloader = get_config("common/optim.py").optimizer
optimizer = get_config("common/optim.py").optimizer
train = get_config("common/train.py").train
accelerator = get_config("common/train.py").accelerator
vae = get_config("common/model.py").vae
unet = get_config("common/model.py").unet
tokenizer = get_config("common/model.py").tokenizer
text_encoder = get_config("common/model.py").text_encoder
noise_scheduler = get_config("common/model.py").noise_scheduler
