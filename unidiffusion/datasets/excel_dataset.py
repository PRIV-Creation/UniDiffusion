from .image_dataset import ImageDataset
from omegaconf import ListConfig
import pandas as pd
import random
from PIL import Image


prompts = [
    "A woman wearing {clothing_item}."
    # "A female model elegantly wearing a {clothing_item}.",
    # "A close-up of a woman dressed in a {clothing_item}.",
    # "A stylish woman showcasing a {clothing_item}.",
    # "An elegant female model posing in a {clothing_item}.",
    # "A casual look featuring a woman in a {clothing_item}.",
    # "A sophisticated female model wearing a {clothing_item}.",
    # "A woman presenting a {clothing_item}.",
    # "A fashion portrait of a female in a {clothing_item}.",
    # "A chic woman modeling a {clothing_item}.",
    # "A confident woman modeling a {clothing_item}.",
    # "A woman in a {clothing_item}, captured in a studio.",
    # "A female model showcasing a {clothing_item}.",
    # "Portrait of a woman wearing a {clothing_item}.",
    # "A full-length image of a woman dressed in a {clothing_item}.",
    # "A lady wearing a {clothing_item}.",
    # "A woman showcasing a {clothing_item}.",
    # "A casual shot of a female wearing a {clothing_item}.",
    # "A woman modeling a {clothing_item}.",
    # "A female fashion model in a {clothing_item}.",
    # "An artistic portrayal of a woman in a {clothing_item}.",
    # "A lifestyle shot of a woman wearing a {clothing_item}.",
    # "A woman in a {clothing_item}, posing elegantly.",
    # "A close-up portrait of a woman in a {clothing_item}.",
    # "A stylish woman in a {clothing_item}.",
    # "An image of a woman dressed in a {clothing_item}.",
    # "A woman wearing a {clothing_item}.",
    # "A fashionable woman in a {clothing_item}.",
    # "A woman modeling a {clothing_item}, looking chic.",
    # "A portrait of a woman in a {clothing_item}.",
    # "A casual pose of a woman in a {clothing_item}.",
    # "A full-body shot of a woman wearing a {clothing_item}.",
    # "A street style look featuring a woman in a {clothing_item}.",
    # "A chic look of a woman in a {clothing_item}.",
    # "A female model in a {clothing_item}.",
    # "A relaxed look of a woman wearing a {clothing_item}.",
    # "A vibrant shot of a woman wearing a {clothing_item}.",
    # "A sophisticated woman in a {clothing_item}.",
    # "A stylish portrait of a woman in a {clothing_item}.",
    # "A laid-back style of a woman in a {clothing_item}.",
    # "A fashion statement by a woman wearing a {clothing_item}.",
    # "A relaxed yet stylish woman in a {clothing_item}.",
    # "A fashionable look of a woman in a {clothing_item}.",
    # "A chic and casual woman in a {clothing_item}.",
    # "A woman in a {clothing_item}, looking fashionable.",
    # "A trendy woman showcasing a {clothing_item}.",
    # "A woman in a {clothing_item}, striking a pose.",
    # "A fashionable woman presenting a {clothing_item}.",
    # "A woman in a {clothing_item}, looking stylish.",
    # "A woman in a {clothing_item}, exuding elegance.",
    # "A woman wearing a {clothing_item}, looking chic and modern."
]


class ExcelDataset(ImageDataset):
    def __init__(self, excel_path="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # read excel
        df = pd.read_excel(excel_path, engine='openpyxl')
        prompt_dict = dict()
        for i, row in df.iterrows():
            store_dict = prompt_dict.get(row['Store'], dict())
            store_dict[row['id']] = row['title'].lower()
            prompt_dict[row['Store']] = store_dict
        self.prompt_dict = prompt_dict

    def get_prompt(self, index):
        image_path = self.image_paths[index % self.num_instance_images]
        info = image_path.split('/')
        try:
            cloth_prompt = self.prompt_dict[info[2]][int(info[3])]
            prompt = random.choice(prompts).format(clothing_item=cloth_prompt)
            return prompt
        except:
            return random.choice(prompts).format(clothing_item="cloth")
