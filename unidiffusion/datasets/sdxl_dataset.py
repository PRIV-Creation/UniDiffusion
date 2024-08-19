from .dataset import BaseDataset
from torchvision import transforms
import random
import numpy as np
from datasets import load_dataset,concatenate_datasets
from torchvision.transforms.functional import crop
import functools
import torch

DATASET_NAME_MAPPING = {
"lambdalabs/naruto-blip-captions": ("image", "text"),
}

def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
                return_dict=False,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()}

def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.cpu()}

class HuggingFaceDataset_SDXL(BaseDataset):

    def __init__(
            self,
            path,
            name,
            cache_dir,
            tokenizer_one,
            tokenizer_two,
            text_encoder_one,
            text_encoder_two,
            vae,
            accelerator,
            train_batch_size,
            image_column="image",
            caption_column="text",
            resolution=512,
            center_crop=False,
            random_flip=True,
            proportion_empty_prompts=0,
            max_train_samples=None,
            seed=None
    ):
        super().__init__()
        resolution = resolution[0]
        vae = vae[0]
        dataset_name = path
        dataset_config_name = name
        cache_dir = cache_dir
        dataset = load_dataset(path, name, cache_dir=cache_dir)
        caption_column = caption_column
        column_names = dataset["train"].column_names
        dataset_columns = DATASET_NAME_MAPPING.get(dataset_name, None)
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        if image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
                )
        # Preprocessing the datasets.
        train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            # image aug
            original_sizes = []
            all_images = []
            crop_top_lefts = []
            for image in images:
                original_sizes.append((image.height, image.width))
                image = train_resize(image)
                if random_flip and random.random() < 0.5:
                    # flip
                    image = train_flip(image)
                if center_crop:
                    y1 = max(0, int(round((image.height - resolution) / 2.0)))
                    x1 = max(0, int(round((image.width - resolution) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (resolution, resolution))
                    image = crop(image, y1, x1, h, w)
                crop_top_left = (y1, x1)
                crop_top_lefts.append(crop_top_left)
                image = train_transforms(image)
                all_images.append(image)

            examples["original_sizes"] = original_sizes
            examples["crop_top_lefts"] = crop_top_lefts
            examples["pixel_values"] = all_images
            return examples
        with accelerator.main_process_first():
            if max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
            dataset = dataset["train"].with_transform(preprocess_train)
        compute_embeddings_fn = functools.partial(
            encode_prompt,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            proportion_empty_prompts=proportion_empty_prompts,
            caption_column=caption_column,
        )
        compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
        with accelerator.main_process_first():
            from datasets.fingerprint import Hasher
            new_fingerprint = Hasher.hash(self)
            new_fingerprint_for_vae = Hasher.hash(vae.__class__)
            train_dataset_with_embeddings = dataset.map(
            compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint
            )
            train_dataset_with_vae = dataset.map(
            compute_vae_encodings_fn,
            batched=True,
            batch_size=train_batch_size,
            new_fingerprint=new_fingerprint_for_vae,
        )
        precomputed_dataset = concatenate_datasets(
            [train_dataset_with_embeddings, train_dataset_with_vae.remove_columns(["image", "text"])], axis=1
        )
        self.precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)

    def __len__(self):
        return len(self.precomputed_dataset)

    def __getitem__(self, index):
        example = self.precomputed_dataset[index]
        if "text" in example and "prompt" not in example:
            example["prompt"] = example["text"]
            example.pop("text")
        return example

    def get_prompt(self, item):
        return self[item]["prompt"]
