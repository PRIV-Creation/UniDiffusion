from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def get_prompt(self, item):
        pass

    def get_placeholders(self):
        pass


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self._length = sum([len(dataset) for dataset in datasets])

        # construct index
        self.dataset_index = []
        self.data_index = []
        for d in datasets:
            for i in range(len(d)):
                self.dataset_index.append(d)
                self.data_index.append(i)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.datasets[self.dataset_index[index]][self.data_index[index]]

    def get_prompt(self, item):
        return self.datasets[self.dataset_index[item]].get_prompt(self.data_index[item])
