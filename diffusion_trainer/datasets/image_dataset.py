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
        resolution=512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.resolution = resolution

        self.image_paths = glob.glob(f'{image_paths}/**/*.png', recursive=True)
        self.image_paths.sort()

        self.num_instance_images = len(self.image_paths)

        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.image_paths[index % self.num_instance_images])

        prompt = f'a photo of <{self.placeholder}>'
        example["pixel_values"] = self.image_transforms(instance_image)
        example["input_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example