from unidiffusion.config import get_config


# get default config
dataloader = get_config("common/dataloader.py").dataloader
checkpoint = get_config("common/common.py").checkpoint
optimizer = get_config("common/optim.py").optimizer
lr_scheduler = get_config("common/scheduler.py").lr_scheduler
train = get_config("common/common.py").train
inference = get_config("common/common.py").inference
inference_pipeline = get_config("common/common.py").inference_pipeline
evaluation = get_config("common/common.py").evaluation
accelerator = get_config("common/common.py").accelerator
vae = get_config("common/model.py").vae
unet = get_config("common/model.py").unet
unet_init = get_config("common/model.py").unet_init
tokenizer = get_config("common/model.py").tokenizer
text_encoder = get_config("common/model.py").text_encoder
noise_scheduler = get_config("common/model.py").noise_scheduler
