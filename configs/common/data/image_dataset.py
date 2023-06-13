import torch
from unidiffusion.datasets import ImageDataset
from unidiffusion.config import LazyCall as L
from unidiffusion.utils.dataloader import collate_fn


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
