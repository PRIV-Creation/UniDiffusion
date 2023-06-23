## Train textual inversion / Dreambooth / LoRA / text-to-image Finetune.
### 1. Textual inversion & Dreambooth
We use `sampels/faces` as example, with additional token ```<face>```.
```bash
accelerate launch scripts/train.py --config-file configs/train/textual_inversion.py
accelerate launch scripts/train.py --config-file configs/train/dreambooth.py
```

We use `image_dataset` in `unidiffusion/datasets/image_dataset` to load images and set a text token in ```a photo of <>```.

```python
dataset = get_config("common/data/image_dataset.py").dataset
dataset.path = 'samples/faces'
dataset.placeholder = None
dataset.inversion_placeholder = '<face>'    # set textual inversion tokens
```
`dataset.inversion_placeholder` indicates additional textual inversion token (we suggest using ```<xxx>``` format), while `dataset.placeholder` means use existing token in tokenizer and will be not trained. 


For training arguments, we set unet and text_encoder 
```python
# textual inversion not set unet.training_args
# set mode to 'lora' to enabled dreambooth_lora
unet.training_args = {
    '': {
        'mode': 'finetune',
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}

text_encoder.training_args = {
    'text_embedding': {
        'initial': True,         # whether to init additional token by their text.
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
```

### 2. LoRA and text-to-image Finetune
We use [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) dataset as example.
```bash
accelerate launch scripts/train.py --config-file configs/train/lora.py
accelerate launch scripts/train.py --config-file configs/train/text_to_image_finetune.py
```

```bash
dataset.path = 'lambdalabs/pokemon-blip-captions'

unet.training_args = {
    '': {
        'mode': 'finetune', # or 'lora'
        'optim_kwargs': {'lr': '${optimizer.lr}'}
    }
}
```
Set `mode` to 'finetune' or 'lora' to enable each finetuning mechanism.