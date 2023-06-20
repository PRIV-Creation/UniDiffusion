from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def get_placeholders(self):
        pass

