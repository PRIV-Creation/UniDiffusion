import torch
from unidiffusion.config import LazyCall as L
from unidiffusion.utils.dataloader import collate_fn,collate_fn_sdxl


dataloader = L(torch.utils.data.DataLoader)(
    collate_fn=collate_fn,
    shuffle=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
)

dataloader_sdxl = L(torch.utils.data.DataLoader)(
    collate_fn=collate_fn_sdxl,
    shuffle=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
)