from unidiffusion.config import get_config


# get default config
dataloader = get_config("data/image_dataset.py").dataloader
optimizer = get_config("optim.py").optimizer
lr_scheduler = get_config("scheduler.py").lr_scheduler
train = get_config("train.py").train
accelerator = get_config("train.py").accelerator
vae = get_config("model.py").vae
unet = get_config("model.py").unet
tokenizer = get_config("model.py").tokenizer
text_encoder = get_config("model.py").text_encoder
noise_scheduler = get_config("model.py").noise_scheduler
