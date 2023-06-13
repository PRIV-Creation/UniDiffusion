# UniDiffusion
[![License](https://img.shields.io/badge/license-apache2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.0-blue.svg)](https://pytorch.org/)

Easily conducting the **Uni**versity of Diffusion based on **Uni**fied pipeline.

## Introduction
UniDiffusion is a toolbox that provides state-of-the-art training and inference algorithms, based on diffusers.

UniDiffusion is built to solve:
- Using all training methods in a unified way. 
- Easy to use, easy to customize, easy to combine, and easy to extend.
- Clean and readable codebase.

**Note:** UniDiffusion is still under development. Some modules are borrowed from other code repositories and have not been tested yet, especially the components that are not enabled by default in the configuration system. We are working hard to improve this project.

## Features
- *Decomposing Training Methods*. UniDiffusion decomposes methods into three dimension: trainable parameters, peft method, and training process. This process facilitates greater ease in the combination and enhancement of the method.
- *Modular Design*. UniDiffusion is designed with a modular architecture. The modular design enables easy implementation of new methods. 
- *Config System*. LazyConfig System for more flexible syntax and cleaner config files.
- *Easy to Use*.
  - The configuration files for all existing methods have been predefined. 
  - Using [accelerate](https://github.com/huggingface/accelerate) to support all distributed training environment. 
  - Using [wandb](https://wandb.ai/) to log all training information.

## Unified Training Pipeline
In UniDiffusion, all training methods are decomposed into three dimensions: trainable parameters, peft method, and training process, which allows we conduct a unified training pipeline with strong config system.

Here is a simple example. In diffusers, training `text-to-image finetune` and `dreambooth` like:
```bash
python train_dreambooth.py --arg1 ......
python train_finetune.py --arg1 ......
```
and combining or adjusting some of methods are difficult (e.g., only training cross attention during dreambooth).

In UniDiffusion, we can easily design our own training arguments in config file:
```python
# text-to-image finetune
unet.training_args = {'*': {'mode': 'finetune'}}
# text-to-image finetune with original lora
unet.training_args = {'*.cross-attention*.(K|V)': {'mode': 'lora'}}
# text-to-image finetune with lora (whole model)
unet.training_args = {'*': {'mode': 'lora'}}
# text-to-image finetune with original lora while finetune residual block with small learning rate
unet.training_args = {'*.cross-attention*.(K|V)': {'mode': 'lora'}, '*.residual*': {'mode': 'finetune', 'lr_mul': 0.1}}

# dreambooth
unet.training_args = {'*': {'mode': 'finetune'}}
text_encoder.training_args = {'*', {'mode': 'finetune'}}
# dreambooth with small lr for text-encoder
unet.training_args = {'*': {'mode': 'finetune'}}
text_encoder.training_args = {'*', {'mode': 'finetune', 'lr_mul': 0.1}}
```
This facilitates easier customization, combination, and enhancement of methods, and also allows for the comparison of similarities and differences between methods through configuration files.

## Installation
- Python 3.10
- Pytorch 2.0 + CUDA11.8
- CUDNN
```bash
conda create -n difftrainer python=3.10
pip install -e requirements.txt
```

## Acknowledge
- Diffusion Trainer is built based on [diffusers](https://github.com/huggingface/diffusers).
- A lot of module design is borrowed from [detectron2](https://github.com/facebookresearch/detectron2) and [detrex](https://github.com/IDEA-Research/detrex).
- Some implementations of methods is borrowed from  [diffusers](https://github.com/huggingface/diffusers) and [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).

## Citation
If you use this toolbox in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

- Citing **UniDiffusion**:

```BibTeX
@misc{priv2022diffusion,
  author =       {Pu Cao, Tianrui Huang, Lu Yang, Qing Song},
  title =        {UniDiffusion},
  howpublished = {\url{https://github.com/PRIV-Creation/UniDiffusion}},
  year =         {2023}
}
```

<details>
<summary> Citation Supported Algorithms </summary>
