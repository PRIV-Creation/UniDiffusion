import glob
from .dataset import BaseDataset
from torchvision import transforms
from PIL import Image


class ImageDataset(BaseDataset):
    def __init__(
        self,
        image_paths,
        tokenizer,
        placeholder,
        inversion_placeholder,
        resolution=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.inversion_placeholder = inversion_placeholder
        self.resolution = resolution

        self.image_paths = glob.glob(f'{image_paths}/**/*.png', recursive=True)
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

    def get_placeholders(self):
        return [self.inversion_placeholder]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.image_paths[index % self.num_instance_images])

        placeholder = self.inversion_placeholder if self.inversion_placeholder is not None else self.placeholder
        prompt = f'a photo of {placeholder}'
        example['prompt'] = prompt
        example["pixel_values"] = self.image_transforms(instance_image)
        example["input_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example