# coding: utf-8

"""
Date: 28-06-2022

Author: Lucas Maison

Train a model from data
"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from GoGRU import GoGRU
from DOCC10 import DOCC10

from copy import deepcopy
from time import time, strftime
from pathlib import Path

# -------------------------------------------------------------------------

def accuracy(inputs, targets):
    """
    Computes the accuracy score
    """
    
    _, predicted = torch.max(inputs, 1)
    r = (predicted == targets)
    r = torch.Tensor.float(r).mean()
    
    return r

def norm_regularization(model, norm_type='nuc'):
    """
    Computes the sum of the norms of the model's weight matrices
    """
    
    weights_norms_sum = 0.0

    for name, weight in model.named_parameters():
        if 'gru.weight' in name:
            weights_norms_sum = weights_norms_sum + torch.linalg.norm(weight, ord=norm_type)
    
    return weights_norms_sum.cpu()


def LRA_loss(inputs, targets, criterion, epoch, lambda_, model, norm):
    """
    Computes the full loss (Cross Entropy + Norm Regularization)
    """
    
    loss = criterion(inputs, targets)

    # increase the weight of the regularization progressively
    epoch_coef = 0.0 if epoch < 5 else ((epoch - 5) / (25 - 5)) if epoch < 25 else 1.0
    reg_coef = lambda_ * epoch_coef

    # computes the norm regularization only if necessary
    if reg_coef > 0.0:
        loss = loss + reg_coef * norm_regularization(model, norm)

    return loss

def perform_LRA(model, target_rank, epoch, time_interval, device):  
    """
    Performs the Low Rank Approximation method by
    updating each of the weight matrices
    """
    
    if epoch and epoch % time_interval == 0:
        modules = {}

        for name, tensor in model.state_dict().items():
            if 'gru.weight' in name:
                U, S, V = torch.linalg.svd(tensor, full_matrices=False)
                Ur, Sr, Vr = U[:, :target_rank], S[:target_rank], V[:target_rank, :]

                modules[name] = torch.matmul(Ur * Sr, Vr).to(device)

        model.load_state_dict(modules, strict=False)

# -------------------------------------------------------------------------

dataset_path = str(Path.home()) + "/datasets/DOCC10/DOCC10_train/"
train_size = 0.8
seed = 91741
bz = 256 # batch size
epochs = 50
lr = 1e-3 # learning rate
lambda_ = 0.0#1e-3
device = "cuda:0"

# read data
X = np.load(dataset_path + "DOCC10_Xtrain_small.npy")
Y_df = pd.read_csv(dataset_path + "DOCC10_Ytrain.csv", index_col=0)
y = Y_df["TARGET"].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# preprocess data
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("y_enc shape:", y_enc.shape)
os.makedirs("pickle/label_encoder/", exist_ok=True)
pickle.dump(le, open("pickle/label_encoder/label_encoder_DOCC10.pkl", "wb"))

# split data
X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y_enc,
                                                  train_size=train_size,
                                                  random_state=seed,
                                                  shuffle=True)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# scale data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_val_std = sc.transform(X_val)
os.makedirs("pickle/standard_scaler/", exist_ok=True)
pickle.dump(sc, open("pickle/standard_scaler/standard_scaler_DOCC10.pkl", "wb"))

# create dataset objects
trainset = DOCC10(X_train_std, y_train, b=len(X_train_std)//bz * bz)
valset = DOCC10(X_val_std, y_val, b=len(X_val_std)//bz * bz)

print("Datasets' length (may be smaller than original data due to last batch dismissal):")
print("Trainset:", len(trainset))
print("Valset:", len(valset))

# create dataloaders
trainloader = DataLoader(trainset, batch_size=bz, shuffle=True)
valloader = DataLoader(valset, batch_size=bz, shuffle=True)

# create model
model = GoGRU(dropout=0.5)
model.to(device)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# define arrays and other variables
train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)
train_accs = np.zeros(epochs)
val_accs = np.zeros(epochs)
trace_norms = np.zeros(epochs)

bestModels = [None]
bestValAccs = [0.0]

# define optimizer, scheduler, criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()

lambda_lr = lambda epoch: max(0.96 ** (epoch - 0), 1e-2)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, verbose=False)
criterion = torch.nn.CrossEntropyLoss()

# start training
for epoch in range(epochs):
    start = time()
    
    ### TRAINING ###
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for train_input, train_target in trainloader:
        train_input = train_input.to(device)
        train_target = train_target.to(device)

        out = model(train_input)
        loss = LRA_loss(out, train_target, criterion, epoch, lambda_, model, 'nuc')
        loss.backward()

        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        
        train_acc += accuracy(out, train_target)

    train_loss /= len(trainloader)
    train_losses[epoch] = train_loss
    
    train_acc /= len(trainloader)
    train_accs[epoch] = train_acc

    scheduler.step()
    
    ### VALIDATION ###
    model.eval()

    val_loss = 0.0
    val_acc = 0.0

    for val_input, val_target in valloader:
        val_input = val_input.to(device)
        val_target = val_target.to(device)

        with torch.no_grad():
            out = model(val_input)
            loss = criterion(out, val_target)

        val_loss += loss.item()
        val_acc += accuracy(out, val_target)

    val_loss /= len(valloader)
    val_losses[epoch] = val_loss
    
    val_acc /= len(valloader)
    val_accs[epoch] = val_acc

    trace_norm = norm_regularization(model)
    trace_norms[epoch] = trace_norm

    print("Epoch %i : train acc %0.3f, val acc %0.3f, trace norm %0.1f"%(epoch, train_acc, val_acc, trace_norm))

    if val_acc > bestValAccs[-1] + 0.5e-3:
        bestValAccs.append(val_acc)
        bestModels.append(deepcopy(model.state_dict()))
        print("\tSaving model checkpoint with val acc %0.4f"%val_acc)

    perform_LRA(model, 20, epoch, 10000, device)
    print("Time elapsed: %0.1fs"%(time()-start))
    print()

print("Best train acc: %0.3f, best val acc: %0.3f (at epoch %i)"%(np.max(train_accs), np.max(val_accs), np.argmax(val_accs)))

# save arrays in order to create figures later
timestr = strftime("%d%m%Y-%H%M%S")
os.makedirs("npy/", exist_ok=True)
np.save("npy/DOCC10_%s_train_losses.npy"%timestr, train_losses)
np.save("npy/DOCC10_%s_train_accs.npy"%timestr, train_accs)
np.save("npy/DOCC10_%s_val_losses.npy"%timestr, val_losses)
np.save("npy/DOCC10_%s_val_accs.npy"%timestr, val_accs)
np.save("npy/DOCC10_%s_trace_norms.npy"%timestr, trace_norms)

# save best checkpoint to disk
os.makedirs("models/DOCC10/", exist_ok=True)
torch.save(bestModels[-1], "models/DOCC10/gogru.pt")
