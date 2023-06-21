from unidiffusion.datasets import ImageDataset
from unidiffusion.config import LazyCall as L


dataset = L(ImageDataset)(
    image_paths='samples/faces',
    placeholder=None,
    inversion_placeholder='<face>'
)
