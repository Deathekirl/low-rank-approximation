# coding: utf-8

"""
Date: 21-07-2022

Author: Lucas Maison

Run a model on test data

DOCC10: ... and outputs submission file for the challenge
SequentialMNIST: ... and create a figures of qualitative results
"""

import os
import pickle
import torch

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from pathlib import Path

device = "cuda:0"
task_name = "SequentialMNIST"

# -------------------------------------------------------------------------

print("Task :", task_name)

if task_name == "DOCC10":
    from DOCC10 import DOCC10
    from GoGRU import GoGRU

    dataset_path = str(Path.home()) + "/datasets/DOCC10/DOCC10_test/"
    submission_filename = "submissions/DOCC10_Ytest_pred_noLRA.csv"
    os.makedirs("submissions/", exist_ok=True)

    # load test data
    X_test = np.load(dataset_path + "DOCC10_Xtest_small.npy")
    print("X_test shape:", X_test.shape)

    # scale data using pickled StandardScaler
    sc = pickle.load(open("pickle/standard_scaler/standard_scaler_DOCC10.pkl", "rb"))
    X_test_std = sc.transform(X_test)

    # create test set
    testset = DOCC10(X_test_std, np.zeros(X_test_std.shape[0]))
    print("Length of test set:", len(testset))

    # create model and load a checkpoint
    model = GoGRU()
    state = torch.load("models/DOCC10/gogru.pt")
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # compute predictions
    predictions = []

    for X, _ in testset:  # can take a long time
        with torch.no_grad():
            out = model(X.unsqueeze(0).to(device))
        predictions.append(torch.argmax(out).item())

    print("Number of predictions:", len(predictions))
    print("Writing predictions to file: %s ..." % submission_filename)

    # write predictions to submission file
    le = pickle.load(open("pickle/label_encoder/label_encoder_DOCC10.pkl", "rb"))
    y_test_pred = le.inverse_transform(predictions)
    Y_test_pred_df = pd.DataFrame(
        data=y_test_pred, index=np.arange(y_test_pred.shape[0]), columns=["TARGET"]
    )
    Y_test_pred_df.to_csv(submission_filename, index_label="ID")
elif task_name == "SequentialMNIST":
    from torchvision.datasets import MNIST
    from GoGRU import GoGRU_sequence
    from SequentialMNIST import SequentialMNIST
    from PIL import Image

    # create test set
    dataset_test = MNIST(
        root=str(Path.home()) + "/datasets", train=False, download=True
    )
    testset = SequentialMNIST(dataset_test)
    print("Length of test set:", len(testset))
    testloader = DataLoader(testset, batch_size=256, shuffle=False, drop_last=True)

    timecode = "21072022-132652"

    # create models
    model = GoGRU_sequence()
    state = torch.load("models/SequentialMNIST/gogru_SequentialMNIST_%s.pt" % timecode)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    # compute test loss
    loss = 0.0
    for top, bottom in testloader:
        top = top.to(device)
        bottom = bottom.to(device)

        with torch.no_grad():
            prediction = model(top)
        mse = torch.nn.MSELoss()(prediction, bottom).item()

        loss += mse
    loss /= len(testloader)
    print("Test loss: %0.3f" % loss)

    arrays = []
    count = 0
    testloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=True)
    # create a figure showing reconstruction artifacts
    for top, bottom in testloader:
        top = top.to(device)
        bottom = bottom.to(device)

        with torch.no_grad():
            prediction = model(top)
        original = torch.cat((top, bottom), axis=0).squeeze().reshape(28, 28)
        reconstructed = torch.cat((top, prediction), axis=0).squeeze().reshape(28, 28)
        merge = torch.cat((original, reconstructed), axis=0)

        unnormalized = torch.relu(merge * 0.3081 + 0.1307) * 255
        array = unnormalized.cpu().detach().numpy().astype(np.uint8)
        arrays.append(array)

        count += 1
        if count == 30:
            break

    image = Image.fromarray(np.concatenate(arrays, axis=1))

    os.makedirs("figures/SequentialMNIST/", exist_ok=True)
    image.save("figures/SequentialMNIST/quality_of_reconstruction_%s.png" % timecode)

print("Done")
