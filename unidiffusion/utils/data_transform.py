import torchvision.transforms as transforms


def aspect_ratio_transform(image, target_height, target_width, flip_prob):
    # calculate the aspect ratio
    specified_aspect_ratio = target_width / target_height

    w, h = image.size
    crop_width = int(h * specified_aspect_ratio)

    # 创建transform
    transform = transforms.Compose([
        transforms.CenterCrop((h, w)),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.Resize((target_height, target_width)),
    ])

    # 应用transform
    return transform(image)
