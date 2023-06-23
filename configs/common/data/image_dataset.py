from unidiffusion.datasets import ImageDataset
from unidiffusion.config import LazyCall as L


dataset = L(ImageDataset)(
    path=None,
    placeholder=None,
    inversion_placeholder=None,
    resolution='${train.resolution}',
)
