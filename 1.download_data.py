# coding: utf-8

"""
Date: 20-07-2022

Author: Lucas Maison

Download MNIST dataset
"""

from urllib.request import urlretrieve
from torchvision.datasets import MNIST
from pathlib import Path

path = str(Path.home()) + "/datasets"

# MNIST
dataset_train = MNIST(root=path, train=True, download=True)
dataset_test = MNIST(root=path, train=False, download=True)

# AugMod
urlretrieve(
    "https://augmod.blob.core.windows.net/augmod/augmod.zip",
    filename=path + "/augmod.zip",
)
