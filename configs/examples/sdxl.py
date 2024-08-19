from configs.common.get_common_config_sdxl import *
dataset = get_config("common/data/sdxl_dataset.py").dataset
train.pretrained_model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
# train.pretrained_model_name_or_path = '/home/yanrupeng/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b'
unet.training_args = {
    r'attn2\.(to_k|to_v)': {
        'mode': 'lora',
        'module_kwargs': {'scale': 1.0, 'rank': 4},
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}