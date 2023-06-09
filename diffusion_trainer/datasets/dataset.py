from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, cfg):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
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
