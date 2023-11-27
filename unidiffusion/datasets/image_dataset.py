import torch
import glob
import random
from .dataset import BaseDataset
from torchvision import transforms
from PIL import Image
from omegaconf import ListConfig


class ImageDataset(BaseDataset):
    def __init__(
        self,
        path,
        tokenizer,
        placeholder,
        inversion_placeholder,
        resolution=512,
        is_training=True,
        drop_prob=0.,
        prompt_template="a photo of {}",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.inversion_placeholder = inversion_placeholder
        self.resolution = resolution
        self.prompt_template = prompt_template

        images = []
        for ext in ['jpg', 'png']:
            images.extend(glob.glob(f'{path}/**/*.{ext}', recursive=True))
        self.image_paths = images
        self.image_paths.sort()

        self.num_instance_images = len(self.image_paths)

        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resolution, antialias=None),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            prompts = [example["prompt"] for example in examples]
            return {"pixel_values": pixel_values, "input_ids": input_ids, "prompt": prompts}

        self.preprocess_train = preprocess_train

        self.drop_prob = drop_prob if is_training else 0.

    def get_placeholders(self):
        if not isinstance(self.inversion_placeholder, (list, ListConfig)):
            return [self.inversion_placeholder]
        else:
            return self.inversion_placeholder

    def __len__(self):
        return self._length

    def get_prompt(self, item):
        placeholder = self.inversion_placeholder if self.inversion_placeholder is not None else self.placeholder
        if isinstance(placeholder, list):
            placeholder = ' '.join(placeholder)
        prompt = self.prompt_template.format(placeholder)
        return prompt

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.image_paths[index % self.num_instance_images])
        if instance_image.mode != 'RGB':
            instance_image = instance_image.convert('RGB')

        if random.random() < self.drop_prob:
            prompt = ""
        else:
            placeholder = self.inversion_placeholder if self.inversion_placeholder is not None else self.placeholder
            if isinstance(placeholder, list):
                placeholder = ' '.join(placeholder)
            prompt = self.prompt_template.format(placeholder)

        example['file_name'] = self.image_paths[index % self.num_instance_images]
        example['prompt'] = prompt
        example["pixel_values"] = self.image_transforms(instance_image)
        if self.tokenizer is None:
            example["input_ids"] = torch.tensor([1])
        else:
            example["input_ids"] = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example