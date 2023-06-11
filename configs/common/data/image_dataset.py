import torch
from diffusion_trainer.datasets import ImageDataset
from diffusion_trainer.config import LazyCall as L
from diffusion_trainer.utils.dataloader import collate_fn


dataloader = L(torch.utils.data.DataLoader)(
    dataset=L(ImageDataset)(
        image_paths='samples/faces',
        placeholder='face',
    ),
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=2,
    num_workers=2,
)
