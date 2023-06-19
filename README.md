<div align="center">

  <img src="assets/temp_logo.png" width="30%">
</div>
<h2 align="center">UniDiffusion</h2>
<p align="center">Navigate the <strong>Uni</strong>verse of <strong>Diffusion</strong> models with <strong>Uni</strong>fied workflow.</p>
<p align="center">
    <a href="https://github.com/PRIV-Creation/UniDiffusion/releases">
        <img alt="release" src="https://img.shields.io/github/v/release/PRIV-Creation/UniDiffusion">
    </a>
    <a href="">
        <img alt="docs" src="https://img.shields.io/badge/docs-Doing-blue">
    </a>
    <a href="https://github.com/PRIV-Creation/UniDiffusion/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/PRIV-Creation/UniDiffusion.svg?color=blue">
    </a>
    <a href="https://github.com/PRIV-Creation/UniDiffusion/">
        <img alt="open issues" src="https://img.shields.io/badge/python-3.10-blue.svg">
    </a>
    <a href="https://github.com/PRIV-Creation/UniDiffusion/">
        <img alt="open issues" src="https://img.shields.io/badge/pytorch-2.0.0-blue.svg">
    </a>
</p>



## Introduction
UniDiffusion is a toolbox that provides state-of-the-art training and inference algorithms, based on diffusers.
UniDiffusion is aimed at researchers and users who wish to deeply customize the training of stable diffusion. We hope that this code repository can provide excellent support for future research and application extensions.

If you also want to implement the following things, have fun with UniDiffusion </summary>
- Train only cross attention (or conv/...) layer.
- Set different lr/weight decay/... for different layers.
- Using or supporting different/new PEFT/PETL methods for different layers and easily merging them, e.g., finetune the convolution layer and update attention with lora.
- Train all parameter in stable diffusion, including unet, vae, text_encoder, and automatically save and load.
- ......

**Note:** UniDiffusion is still under development. Some modules are borrowed from other code repositories and have not been tested yet, especially the components that are not enabled by default in the configuration system. We are working hard to improve this project.

## Features
- **Modular Design**. UniDiffusion is designed with a modular architecture. The modular design enables easy implementation of new methods. 
- **Config System**. LazyConfig System for more flexible syntax and cleaner config files.
- **Easy to Use**.
  - Distributed Training: Using [accelerate](https://github.com/huggingface/accelerate) to support all distributed training environment. 
  - Experiment Tracker: Using [wandb](https://wandb.ai/) to log all training information.
- **Unified**:
  - [Training Workflow](#unified-training-workflow)
  - [Trainable Parameter/Module Selection](#regular-matching-for-module-selection)
  - [PEFT/PETL Methods Support](#powerful-support-for-peftpetl-methods)

### Unified Training Workflow
In UniDiffusion, all training methods are decomposed into three dimensions
- Trainable parameters: which layer or which module will be updated?
- PEFT/PETL method: how to update them? E.g., finetune, low-rank adaption, adapter, etc.
- Training process: default to diffuion-denoising, which can be extended like XTI.

It allows we conduct a unified training pipeline with strong config system.

<details>
<summary> Example for difference in training pipeline </summary>
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
</details>

### Regular Matching for Module Selection
In UniDiffusion, we provide a regular matching system for module selection. It allows us to select modules by regular matching. See [Module Regular Matching](docs/ModuleRegularMatching.md) for more details.

### Powerful Support for PEFT/PETL Methods
We provide a powerful support for PEFT/PETL methods. See [PEFT/PETL Methods](docs/PEFT.md) for more details.

## Installation
1. Install prerequisites
- Python 3.10
- Pytorch 2.0 + CUDA11.8
- CUDNN
2. Install requirements
```bash
pip install -e requirements.txt
```
3. Configuring accelerate and wandb
```bash
accelerate config
wandb login
```
## Getting Started

## Model Zoo
<details open>
<summary> Supported Personalization Methods</summary>

- [x] [text-to-image finetune](configs/train/text_to_image_finetune.py)
- [x] [dreambooth](configs/train/text_to_image_finetune.py)
- [x] [lora](configs/train/text_to_image_lora.py)
- [x] [textual inversion](configs/train/textual_inversion.py)
- [ ] XTI
- [ ] Custom Diffusion

*Note:* Personalization methods are decomposes in trainable parameters, PEFT/PETL methods, and training process in UniDiffusion. See config file for more details.
</details>
<details open>
<summary> Supported PEFT/PETL Methods</summary>

- [x] [finetune](unidiffusion/peft/finetune.py)
- [x] [lora](unidiffusion/peft/lora.py)
- [ ] RepAdapter
</details>

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
