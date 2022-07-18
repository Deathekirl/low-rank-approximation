# coding: utf-8

"""
Date: 29-06-2022

Author: Lucas Maison

Run a model on test data and outputs submission file for the challenge
"""

import os
import pickle
import torch

import numpy as np
import pandas as pd

from DOCC10 import DOCC10
from GoGRU import GoGRU

from pathlib import Path

dataset_path = str(Path.home()) + "/datasets/DOCC10/DOCC10_test/"
submission_filename = "submissions/DOCC10_Ytest_pred_nrhlra_2.csv"
os.makedirs("submissions/", exist_ok=True)

device = "cuda:0"

# -------------------------------------------------------------------------

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

for X, _ in testset: # can take a long time
    out = model(X.unsqueeze(0).to(device))
    predictions.append(torch.argmax(out).item())

print("Number of predictions:", len(predictions))
print("Writing predictions to file: %s ..."%submission_filename)

# write predictions to submission file
le = pickle.load(open("pickle/label_encoder/label_encoder_DOCC10.pkl", "rb"))
y_test_pred = le.inverse_transform(predictions)
Y_test_pred_df = pd.DataFrame(data=y_test_pred, index=np.arange(y_test_pred.shape[0]), columns=['TARGET'])
Y_test_pred_df.to_csv(submission_filename, index_label='ID')

print("Done")
