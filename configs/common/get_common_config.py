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

# common_config = get_config("common/common.py")
# model_config = get_config("common/model.py")
# dataloader = get_config("common/dataloader.py").dataloader
# checkpoint = common_config.checkpoint
# optimizer = get_config("common/optim.py").optimizer
# lr_scheduler = get_config("common/scheduler.py").lr_scheduler
# train = common_config.train
# inference = common_config.inference
# inference_pipeline = common_config.inference_pipeline
# evaluation = common_config.evaluation
# accelerator = common_config.accelerator
# vae = model_config.vae
# unet = model_config.unet
# unet_init = model_config.unet_init
# tokenizer = model_config.tokenizer
# text_encoder = model_config.text_encoder
# noise_scheduler = model_config.noise_scheduler