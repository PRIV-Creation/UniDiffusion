from torch.utils.data import Dataset


class BaseDataset(Dataset):
    pass

class TXTDataset(BaseDataset):
    def __init__(self, cfg):
        super(TXTDataset, self).__init__(cfg)

        self.txt_path = cfg.txt_path
        self.txt_list = self.load_txt()

    def load_txt(self):
        pass

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, item):
        pass


class ImageDataset(BaseDataset):
    def __init__(self, cfg):
        super(ImageDataset, self).__init__(cfg)

        self.image_path = cfg.image_path
        self.image_list = self.load_image()

    def load_image(self):
        pass

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        pass