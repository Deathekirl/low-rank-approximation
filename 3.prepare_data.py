# coding: utf-8

"""
Date: 28-06-2022

Author: Lucas Maison

Prepares DOCC10 data by cropping audio samples from 8192 to 256
"""

import numpy as np
from pathlib import Path

dataset_path = str(Path.home()) + "/datasets/DOCC10"

for split in ("train", "test"):
    print("Loading %s split" % split)

    # read data
    X = np.load(dataset_path + "/DOCC10_%s/DOCC10_X%s.npy" % (split, split))

    print("X_%s shape:" % split, X.shape)

    # array of cropped data
    X_small = np.zeros_like(X, shape=(X.shape[0], 256))

    print("X_%s_small shape:" % split, X_small.shape)

    # iterate over audio samples
    for i in range(X.shape[0]):
        signal = X[i]

        index = 8192 // 2
        X_small[i] = signal[index - 128 : index + 128]

    np.save(dataset_path + "/DOCC10_%s/DOCC10_X%s_small.npy" % (split, split), X_small)
    print("Saving: done")
