import torch
from unidiffusion.config import LazyCall as L
from unidiffusion.utils.dataloader import collate_fn


dataloader = L(torch.utils.data.DataLoader)(
    shuffle=True,
    batch_size=2,
    num_workers=2,
)
