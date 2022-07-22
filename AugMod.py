# coding: utf-8

"""
Date: 22-07-2022

Author: Lucas Maison

Defines classes and functions specific to the AugMod dataset
"""

import torch
from torch.utils.data import Dataset

import numpy as np

from h5py import File
from sklearn.model_selection import train_test_split


class AugMod(Dataset):
    def __init__(self, X, y, a=0, b=None):
        self.X = torch.from_numpy(X)[a:b, :, :]
        self.y = torch.from_numpy(y)[a:b]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def read_augmod(fname, train_size, seed):
    """
    Open Augmod dataset
    """

    data = dict()
    with File(fname, "r") as f:
        data["classes"] = [c.decode() for c in f["classes"]]
        data["signals"] = np.array(f["signals"])
        data["modulations"] = np.array(f["modulations"])
        data["snr"] = np.array(f["snr"])
        data["frequency_offsets"] = np.array(f["frequency_offsets"])

    signals = data["signals"]
    targets = data["modulations"]

    print("X shape:", signals.shape)
    signals = signals.transpose((0, 2, 1))
    print("X shape after transpose:", signals.shape)

    print("y shape:", targets.shape)

    norm = np.sqrt(np.mean(signals**2, axis=(1, 2), keepdims=True))
    signals /= norm

    signals_train, signals_val, targets_train, targets_val = train_test_split(
        signals, targets, train_size=train_size, random_state=seed, shuffle=True
    )

    signals_val, signals_test, targets_val, targets_test = train_test_split(
        signals_val, targets_val, train_size=0.5, random_state=seed, shuffle=True
    )

    print("X_train shape:", signals_train.shape)
    print("y_train shape:", targets_train.shape)
    print("X_val shape:", signals_val.shape)
    print("y_val shape:", targets_val.shape)
    print("X_test shape:", signals_test.shape)
    print("y_test shape:", targets_test.shape)

    return signals_train, signals_val, targets_train, targets_val
