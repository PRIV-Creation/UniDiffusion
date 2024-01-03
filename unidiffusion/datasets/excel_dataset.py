from .image_dataset import ImageDataset
from omegaconf import ListConfig
import pandas as pd
import random
from PIL import Image


prompts = [
    "A female model wearing a {clothing_item}, standing against a simple background.",
    "Portrait of a woman dressed in a {clothing_item}, with a neutral colored background.",
    "A stylish woman showcasing a {clothing_item}, with a minimalistic background setting.",
    "An elegant female model posing in a {clothing_item}, with a plain and subtle background.",
    "A casual look featuring a woman in a {clothing_item}, against a simple, uncluttered background.",
    "A sophisticated female model wearing a {clothing_item}, with a basic and elegant backdrop.",
    "A close-up of a woman wearing a {clothing_item}, emphasizing the outfit against a plain background.",
    "A woman presenting a {clothing_item}, standing with a straightforward, clean background.",
    "A fashion portrait of a female in a {clothing_item}, with a very simple and understated background.",
    "A chic woman modeling a {clothing_item}, against a backdrop that is plain and non-distracting."
    "A confident woman modeling a {clothing_item}, with a clean, unobtrusive background.",
    "A woman in a {clothing_item}, captured in a studio with a neutral backdrop.",
    "A female model elegantly showcasing a {clothing_item}, against a simple studio background.",
    "Portrait of a woman wearing a {clothing_item}, with a soft, muted background.",
    "A full-length image of a woman dressed in a {clothing_item}, with a plain background.",
    "A lady wearing a {clothing_item}, presented against a backdrop with minimal textures.",
    "A woman showcasing a {clothing_item}, with a backdrop that enhances the outfit.",
    "A casual shot of a female wearing a {clothing_item}, with a background that's simple and unassuming.",
    "A woman modeling a {clothing_item}, standing in front of a subtly patterned background.",
    "A female fashion model in a {clothing_item}, against a background that's plain yet elegant.",
    "An artistic portrayal of a woman in a {clothing_item}, with a very basic background.",
    "A lifestyle shot of a woman wearing a {clothing_item}, with a background that’s clean and uncomplicated.",
    "A woman in a {clothing_item}, posing against a background that's understated yet stylish.",
    "A close-up portrait of a woman in a {clothing_item}, with a minimal and soft-colored background.",
    "A stylish woman in a {clothing_item}, with a background that complements without distracting.",
    "An image of a woman dressed in a {clothing_item}, with a background that's simple and harmonious.",
    "A female model wearing a {clothing_item}, against a backdrop that is unadorned yet chic.",
    "A woman showcasing a {clothing_item}, with a background that is plain but tasteful.",
    "A casual, everyday look of a woman in a {clothing_item}, with an uncomplicated and clean background.",
    "A woman in a {clothing_item}, standing against a background that's simple and effective.",
    "A fashion-forward woman wearing a {clothing_item}, with a background that's basic but stylish.",
    "A woman modeling a {clothing_item}, with a background that provides a soft contrast.",
    "A portrait of a woman in a {clothing_item}, with a background that's modest yet appealing.",
    "A woman wearing a {clothing_item}, with a backdrop that's understated and complements the outfit.",
    "A fashionable woman in a {clothing_item}, with a background that's plain and facilitates focus on the outfit.",
    "An elegant shot of a woman wearing a {clothing_item}, against a simple, effective backdrop.",
    "A woman dressed in a {clothing_item}, with a background that's neat and doesn't distract.",
    "A trendy woman showcasing a {clothing_item}, with a background that's simple and complements the style.",
    "A woman in a {clothing_item}, with a background that's clear and allows the outfit to stand out.",
    "A casual pose of a woman in a {clothing_item}, with a background that's uncluttered and straightforward.",
    "A full-body shot of a woman wearing a {clothing_item}, against a backdrop that's simple and unobtrusive.",
    "A street style look featuring a woman in a {clothing_item}, with a plain and urban background.",
    "A woman wearing a {clothing_item}, with a background that's minimalistic and modern.",
    "A chic look of a woman in a {clothing_item}, with a background that's subtle yet fashionable.",
    "A female model in a {clothing_item}, with a background that's clean and allows the outfit to shine.",
    "A relaxed look of a woman wearing a {clothing_item}, with an uncomplicated and serene background.",
    "A woman in a {clothing_item}, with a background that's simple, providing a calm setting.",
    "A vibrant shot of a woman wearing a {clothing_item}, with a background that's plain but effective.",
    "A sophisticated woman in a {clothing_item}, with a background that's understated and classy.",
    "A female model wearing a {clothing_item}, with a background that's simple, complementing the fashion.",
    "A stylish portrait of a woman in a {clothing_item}, with a background that's unassuming yet elegant.",
    "A woman showcasing a {clothing_item}, with a background that's basic, enhancing the attire.",
    "A laid-back style of a woman in a {clothing_item}, with a background that's plain and relaxed.",
    "A fashion statement by a woman wearing a {clothing_item}, against a background that's simple and chic.",
    "A relaxed yet stylish woman in a {clothing_item}, with a background that's straightforward and clean.",
    "A woman in a {clothing_item}, with a background that's unadorned but complements the look.",
    "A woman wearing a {clothing_item}, against a background that's simple and provides a neutral setting.",
    "A fashionable look of a woman in a {clothing_item}, with a background that's simple and focuses on the outfit.",
    "A chic and casual woman in a {clothing_item}, with a background that's plain and unobtrusive."
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
