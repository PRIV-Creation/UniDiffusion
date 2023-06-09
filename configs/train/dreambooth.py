from detrex.config import get_config
from .models.dino_r50 import model

# get default config
dataloader = get_config("common/data/coco_vitdet.py").dataloader
optimizer = get_config("common/optim.py").optimizer

optimizer.optimizer = "AdamW"
opitmizer.lr = 2e-7