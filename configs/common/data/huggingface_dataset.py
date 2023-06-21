from datasets import load_dataset
from unidiffusion.config import LazyCall as L


dataset = L(load_dataset)(
    path="lambdalabs/pokemon-blip-captions",
    name="",
    cache_dir="data/",
)
