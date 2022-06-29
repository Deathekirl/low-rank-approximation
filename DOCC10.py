# coding: utf-8

"""
Date: 28-06-2022

Author: Lucas Maison

Defines classes specific to the DOCC10 dataset
"""

import torch
from torch.utils.data import Dataset

class DOCC10(Dataset):
    def __init__(self, X, y, a=0, b=None):
        self.X = torch.from_numpy(X)[a:b, :, None]
        self.y = torch.from_numpy(y)[a:b]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]