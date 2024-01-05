import torchvision.transforms as transforms


def aspect_ratio_transform(image, target_height, target_width, flip_prob):
    # calculate the aspect ratio
    specified_aspect_ratio = target_width / target_height

    w, h = image.size
    crop_width = int(h * specified_aspect_ratio)

    # 创建transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((h, crop_width)),
        transforms.RandomHorizontalFlip(p=flip_prob),
        transforms.Resize((target_height, target_width)),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 应用transform
    return transform(image)
