# coding: utf-8

"""
Date: 19-07-2022

Author: Lucas Maison

Defines classes specific to the SequentialMNIST dataset
"""

from torch.utils.data import Dataset


class SequentialMNIST(Dataset):
    def __init__(self, dataset, a=0, b=None):
        data = (((dataset.data / 255) - 0.1307) / 0.3081).flatten(1, 2).unsqueeze(2)
        self.X = data[a:b, : 28 * 14, :]
        self.Y = data[a:b, 28 * 14 :, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
