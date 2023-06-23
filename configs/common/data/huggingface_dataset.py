from unidiffusion.datasets.huggingface_dataset import HuggingFaceDataset
from unidiffusion.config import LazyCall as L


dataset = L(HuggingFaceDataset)(
    path="lambdalabs/pokemon-blip-captions",
    name="",
    cache_dir="data/",
    resolution='${train.resolution}',
)
