from diffusion_trainer.config import get_config


# get default config
dataloader = get_config("common/data/txt_dataset.py").optimizer
optimizer = get_config("common/optim.py").optimizer

optimizer.optimizer = "AdamW"
optimizer.lr = 2e-7