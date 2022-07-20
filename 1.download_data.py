# coding: utf-8

"""
Date: 19-07-2022

Author: Lucas Maison

Download MNIST dataset
"""

from torchvision.datasets import MNIST
from pathlib import Path

path = str(Path.home()) + "/datasets"

dataset_train = MNIST(root=path, train=True, download=True)
dataset_test = MNIST(root=path, train=False, download=True)
