from unidiffusion.datasets.dreambooth_dataset import DreamBoothDataset, collate_fn
from unidiffusion.config import LazyCall as L

dataset = L(DreamBoothDataset)(
    instance_data_root=None,
    instance_prompt=None,
    tokenizer=None,
    class_data_root=None,
    class_prompt=None,
    class_num=None,
    size=512,
    center_crop=False,
    encoder_hidden_states=None,
    instance_prompt_encoder_hidden_states=None,
    tokenizer_max_length=None,
)
