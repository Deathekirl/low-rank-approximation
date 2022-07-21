# coding: utf-8

"""
Date: 20-07-2022

Author: Lucas Maison

Defines classes and functions specific to the DOCC10 dataset
"""

import os
import pickle
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DOCC10(Dataset):
    def __init__(self, X, y, a=0, b=None):
        self.X = torch.from_numpy(X)[a:b, :, None]
        self.y = torch.from_numpy(y)[a:b]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_DOCC10_data(dataset_path, train_size, seed, load_from_pickle=False):
    # read data
    X = np.load(dataset_path + "DOCC10_Xtrain_small_bis.npy")
    Y_df = pd.read_csv(dataset_path + "DOCC10_Ytrain.csv", index_col=0)
    y = Y_df["TARGET"].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # preprocess data
    if load_from_pickle:
        le = pickle.load(open("pickle/label_encoder/label_encoder_DOCC10.pkl", "rb"))
        y_enc = le.transform(y)
    else:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        os.makedirs("pickle/label_encoder/", exist_ok=True)
        pickle.dump(le, open("pickle/label_encoder/label_encoder_DOCC10.pkl", "wb"))
    print("y_enc shape:", y_enc.shape)

    # split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, train_size=train_size, random_state=seed, shuffle=True
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    # scale data
    if load_from_pickle:
        sc = pickle.load(open("pickle/standard_scaler/standard_scaler_DOCC10.pkl", "rb"))
        X_train_std = sc.transform(X_train)
    else:
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)

        os.makedirs("pickle/standard_scaler/", exist_ok=True)
        pickle.dump(sc, open("pickle/standard_scaler/standard_scaler_DOCC10_bis.pkl", "wb"))
    X_val_std = sc.transform(X_val)

    return X_train_std, y_train, X_val_std, y_val
