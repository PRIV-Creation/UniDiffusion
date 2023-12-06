from configs.common.get_common_config import *


dataset = get_config("common/data/huggingface_dataset.py").dataset

train.output_dir = 'experiments/pokemon/lora'
dataset.path = "lambdalabs/pokemon-blip-captions"
train.pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'

inference.inference_iter = 200

evaluation.evaluation_iter = 200
evaluation.total_num = 50
evaluation.evaluator.inception_score.enabled = True
evaluation.evaluator.clip_score.enabled = True

unet.training_args = {
    # update all KV of cross attention by lora
    r'attn2\.(to_k|to_v)': {
        'mode': 'lora',
        'module_kwargs': {'scale': 1.0, 'rank': 4},
        'optim_kwargs': {'lr': '${optimizer.lr}'},
    },
}