# coding: utf-8

"""
Date: 20-07-2022

Author: Lucas Maison

Train a model from data
"""

import os

import numpy as np

import torch
from torch.utils.data import DataLoader

from copy import deepcopy
from time import time, strftime
from pathlib import Path

# -------------------------------------------------------------------------


def accuracy(inputs, targets):
    """
    Computes the accuracy score
    """

    _, predicted = torch.max(inputs, 1)
    r = predicted == targets
    r = torch.Tensor.float(r).mean()

    return r


def norm_regularization(model, norm_type="nuc"):
    """
    Computes the sum of the norms of the model's weight matrices
    """

    weights_norms_sum = 0.0

    for name, weight in model.named_parameters():
        if "gru.weight" in name:
            weights_norms_sum = weights_norms_sum + torch.linalg.norm(
                weight, ord=norm_type
            )

    return weights_norms_sum.cpu()


def LRA_loss(inputs, targets, criterion, epoch, lambda_, model, norm, epoch_a, epoch_b):
    """
    Computes the full loss : (Cross Entropy / MSE) + Norm Regularization
    """

    loss = criterion(inputs, targets)

    # increase the weight of the regularization progressively
    epoch_coef = (
        0.0
        if epoch < epoch_a
        else ((epoch - epoch_a) / (epoch_b - epoch_a))
        if epoch < epoch_b
        else 1.0
    )
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
            if "gru.weight" in name:
                U, S, V = torch.linalg.svd(tensor, full_matrices=False)
                Ur, Sr, Vr = U[:, :target_rank], S[:target_rank], V[:target_rank, :]

                modules[name] = torch.matmul(Ur * Sr, Vr).to(device)

        model.load_state_dict(modules, strict=False)


# -------------------------------------------------------------------------

device = "cuda:0"
seed = 91741
train_size = 0.8
bz = 256  # batch size
task_name = "SequentialMNIST"

if task_name == "DOCC10":
    from GoGRU import GoGRU
    from DOCC10 import DOCC10, load_DOCC10_data

    # task parameters
    dataset_path = str(Path.home()) + "/datasets/DOCC10/DOCC10_train/"
    epochs = 50
    lr = 1e-3  # learning rate
    lambda_ = 0.0  # 1e-3
    target_rank = 20
    time_interval = 10000  # cancels HLRA if greater than the number of epochs
    epoch_a, epoch_b = 5, 25
    criterion = torch.nn.CrossEntropyLoss()
    classificationTask = True
    regressionTask = False

    # create dataset objects
    X_train_std, y_train, X_val_std, y_val = load_DOCC10_data(
        dataset_path, train_size, seed
    )
    trainset = DOCC10(X_train_std, y_train)
    valset = DOCC10(X_val_std, y_val)

    # create model
    model = GoGRU(dropout=0.5)
elif task_name == "SequentialMNIST":
    from torchvision.datasets import MNIST
    from GoGRU import GoGRU_sequence
    from SequentialMNIST import SequentialMNIST

    # task parameters
    epochs = 150
    lr = 1e-2
    lambda_ = 0.0  # 1e-4
    target_rank = 40
    time_interval = 10000  # 20 # cancels HLRA if greater than the number of epochs
    epoch_a, epoch_b = 10, 120
    criterion = torch.nn.MSELoss()
    classificationTask = False
    regressionTask = True

    # create dataset objects
    dataset_train = MNIST(
        root=str(Path.home()) + "/datasets", train=True, download=True
    )
    length = int(len(dataset_train) * train_size) // bz * bz
    trainset = SequentialMNIST(dataset_train, b=length)
    valset = SequentialMNIST(
        dataset_train, a=length
    )  # last incomplete batch will be dropped, see dataloaders

    # create models
    model = GoGRU_sequence(
        bidirectional=True, hidden_size=100, num_layers=2, dropout=0.2
    )
else:
    print("Unknown task")
    exit(1)

print(
    "Datasets' length (may be smaller than original data due to last batch dismissal):"
)
print("Trainset:", len(trainset))
print("Valset:", len(valset))

# create dataloaders
trainloader = DataLoader(trainset, batch_size=bz, shuffle=True, drop_last=True)
valloader = DataLoader(valset, batch_size=bz, shuffle=True, drop_last=True)

# move model to device
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
bestValLosses = [10000.0]

assert classificationTask == (not regressionTask)

# define optimizer, scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()

def lambda_lr(epoch): return max(0.96 ** (epoch - 0), 1e-2)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, verbose=False)

# start training
for epoch in range(epochs):
    start = time()

    # ----- TRAINING -----
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for train_input, train_target in trainloader:
        train_input = train_input.to(device)
        train_target = train_target.to(device)

        out = model(train_input)
        loss = LRA_loss(
            out, train_target, criterion, epoch, lambda_, model, "nuc", epoch_a, epoch_b
        )
        loss.backward()

        train_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if classificationTask:
            train_acc += accuracy(out, train_target)

    train_loss /= len(trainloader)
    train_losses[epoch] = train_loss

    train_acc /= len(trainloader)
    train_accs[epoch] = train_acc

    scheduler.step()

    # ----- VALIDATION -----
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

        if classificationTask:
            val_acc += accuracy(out, val_target)

    val_loss /= len(valloader)
    val_losses[epoch] = val_loss

    val_acc /= len(valloader)
    val_accs[epoch] = val_acc

    trace_norm = norm_regularization(model)
    trace_norms[epoch] = trace_norm

    if classificationTask:
        print(
            "Epoch %i : train acc %0.3f, val acc %0.3f, trace norm %0.1f"
            % (epoch, train_acc, val_acc, trace_norm)
        )

        if val_acc > bestValAccs[-1] + 0.5e-3:
            bestValAccs.append(val_acc)
            bestModels.append(deepcopy(model.state_dict()))
            print("\tSaving model checkpoint with val acc %0.4f" % val_acc)

    if regressionTask:
        print(
            "Epoch %i : train loss %0.3f, val loss %0.3f, trace norm %0.1f"
            % (epoch, train_loss, val_loss, trace_norm)
        )

        if val_loss < bestValLosses[-1] - 0.5e-3:
            bestValLosses.append(val_loss)
            bestModels.append(deepcopy(model.state_dict()))
            print("\tSaving model checkpoint with val loss %0.4f" % val_loss)

    perform_LRA(model, target_rank, epoch, time_interval, device)
    print("Time elapsed: %0.1fs" % (time() - start))
    print()

if classificationTask:
    print(
        "Best train acc: %0.3f, best val acc: %0.3f (at epoch %i)"
        % (np.max(train_accs), np.max(val_accs), np.argmax(val_accs))
    )

if regressionTask:
    print(
        "Best train loss: %0.3f, best val loss: %0.3f (at epoch %i)"
        % (np.min(train_losses), np.min(val_losses), np.argmin(val_losses))
    )

# save arrays in order to create figures later
timestr = strftime("%d%m%Y-%H%M%S")
os.makedirs("npy/", exist_ok=True)
np.save("npy/%s_%s_train_losses.npy" % (task_name, timestr), train_losses)
np.save("npy/%s_%s_val_losses.npy" % (task_name, timestr), val_losses)

if classificationTask:
    np.save("npy/%s_%s_train_accs.npy" % (task_name, timestr), train_accs)
    np.save("npy/%s_%s_val_accs.npy" % (task_name, timestr), val_accs)
np.save("npy/%s_%s_trace_norms.npy" % (task_name, timestr), trace_norms)

# save best checkpoint to disk
os.makedirs("models/%s/" % task_name, exist_ok=True)
torch.save(bestModels[-1], "models/%s/gogru_%s_%s.pt" % (task_name, task_name, timestr))
