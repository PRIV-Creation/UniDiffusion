from unidiffusion.datasets.sdxl_dataset import HuggingFaceDataset_SDXL
from unidiffusion.config import LazyCall as L


dataset = L(HuggingFaceDataset_SDXL)(
    # path="lambdalabs/pokemon-blip-captions",
    path = "samples/tlpwithprompt",
    name="",
    cache_dir="data/",
    resolution='${train.resolution}',
)
