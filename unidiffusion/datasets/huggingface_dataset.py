from .dataset import BaseDataset
from torchvision import transforms
import random
import numpy as np
from datasets import load_dataset


def tokenize_captions(examples, tokenizer, caption_column, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


class HuggingFaceDataset(BaseDataset):
    def __init__(
            self,
            path,
            name,
            cache_dir,
            tokenizer,
            image_column="image",
            caption_column="text",
            resolution=512,
            center_crop=False,
            random_flip=True
    ):
        super().__init__()
        self.dataset = load_dataset(path, name, cache_dir=cache_dir)["train"]
        self.tokenizer = tokenizer
        self.caption_column = caption_column

        train_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column, is_train=True)
            return examples

        self.dataset = self.dataset.with_transform(preprocess_train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        if "text" in example and "prompt" not in example:
            example["prompt"] = example["text"]
            example.pop("text")
        return example

    def get_prompt(self, item):
        return self[item]["prompt"]
