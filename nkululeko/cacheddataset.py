import torch
import numpy as np


class CachedDataset(torch.utils.data.Dataset):
    def __init__(
        self, df, features, target_column, transform=None, target_transform=None
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.indices = list(self.df.index)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        signal = np.load(self.features.loc[index, "filename"])
        target = self.df.loc[index, self.target_column]
        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if isinstance(self.target_column, list):
            target = tuple(zip(target, target.index))
        return signal, target
