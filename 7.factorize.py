#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 21-07-2022

Author: Lucas Maison

Choose the best ranks for the different matrices of the model
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from time import time

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 16,
    }
)


def accuracy(inputs, targets):
    """
    Computes the accuracy score
    """

    _, predicted = torch.max(inputs, 1)
    r = predicted == targets
    r = torch.Tensor.float(r).mean()

    return r


def valid_accuracy(model, new_weights, original_weights=None):
    if original_weights is not None:
        model.load_state_dict(original_weights)

    model.load_state_dict(new_weights, strict=False)
    model = model.to(device)
    model.eval()

    start = time()

    valid_accs = []
    valid_losses = []

    with torch.no_grad():
        for valid_input, valid_target in valloader:
            valid_input = valid_input.to(device)
            valid_target = valid_target.to(device)

            out = model(valid_input)

            if classificationTask:
                valid_accs.append(accuracy(out, valid_target).cpu())
            valid_losses.append(criterion(out, valid_target).item())

    valid_acc = np.mean(valid_accs)
    valid_loss = np.mean(valid_losses)

    print("Acc %0.3f Loss %0.3f" % (valid_acc, valid_loss))
    print("Elapsed: %0.1fs" % (time() - start))

    return valid_acc, valid_loss


def factorize_using_same_rank_for_all_matrices(state, ranks, threshold, ih_only=False):
    """
    This function takes a model and compute its performance when factorizing
    its matrices at different ranks.
    """

    loss_by_rank = []
    acc_by_rank = []
    memory_by_rank = []
    affected_matrices_by_rank = []

    # iterate over list of ranks
    for rank in ranks:
        print("Working with rank %i" % rank)
        modules = {}

        memory = 0
        affected_matrices = 0

        # compute SVD of weight matrices
        for name, tensor in state.items():
            if (
                "gru.weight" in name
                and rank < min(tensor.shape)
                and (not ih_only or "_ih_" in name)
            ):
                U, S, V = torch.linalg.svd(tensor, full_matrices=False)
                Ur, Sr, Vr = U[:, :rank], S[:rank], V[:rank, :]

                newtensor = torch.matmul(Ur * Sr, Vr)
                error = torch.linalg.norm(newtensor - tensor).item()
                print("For matrix %s, the LRA error is: %0.1f" % (name, error))

                if error < threshold:
                    modules[name] = newtensor.to(device)

                    memory += (
                        Ur.numel() * 4 + Vr.numel() * 4
                    )  # ignoring Sr because it can be stored with Ur
                    affected_matrices += 1
                else:
                    modules[name] = tensor.to(device)
                    memory += tensor.numel() * 4
            else:
                modules[name] = tensor.to(device)
                memory += tensor.numel() * 4

        memory_by_rank.append(memory)
        affected_matrices_by_rank.append(affected_matrices)

        # do a forward pass over the dataset
        valid_acc, valid_loss = valid_accuracy(model, modules, state)

        loss_by_rank.append(valid_loss)
        acc_by_rank.append(valid_acc)

    return loss_by_rank, acc_by_rank, memory_by_rank, affected_matrices_by_rank


def find_optimal_rank_for_each_matrix(state, ranks, perf_thresholds, modelname):
    """
    This function takes a model and compute the optimal rank for each matrix,
    that is, the lowest rank achievable while still respecting precision targets (p1, p2, p3)
    """

    svd_error_by_rank_by_matrix = {}
    loss_by_rank_by_matrix = {}
    accuracy_by_rank_by_matrix = {}

    rank_for_p1_by_matrix = {}  # almost no change
    rank_for_p2_by_matrix = {}  # 0.5% change
    rank_for_p3_by_matrix = {}  # 1% change

    # compute SVD of weight matrices
    for name, tensor in state.items():
        if "gru.weight" in name and min(tensor.shape) > 1:
            print("\nWorking on %s" % name)

            modules = {}
            svd_error_by_rank = []
            loss_by_rank = []
            accuracy_by_rank = []

            U, S, V = torch.linalg.svd(tensor, full_matrices=False)

            go = True

            for rank in ranks:
                # this condition ensures compression
                if go and rank < tensor.shape[0] * tensor.shape[1] / sum(tensor.shape):
                    print("Rank %i" % rank)

                    Ur, Sr, Vr = U[:, :rank], S[:rank], V[:rank, :]
                    newtensor = torch.matmul(Ur * Sr, Vr)

                    error = torch.linalg.norm(newtensor - tensor).item()
                    svd_error_by_rank.append(error)

                    modules[name] = newtensor.to(device)

                    # do a forward pass over the dataset
                    valid_acc, valid_loss = valid_accuracy(model, modules, state)

                    loss_by_rank.append(valid_loss)
                    accuracy_by_rank.append(valid_acc)

                    if (
                        (classificationTask and valid_acc >= perf_thresholds[2] or not classificationTask and valid_loss <= perf_thresholds[2])
                        and name not in rank_for_p3_by_matrix
                    ):
                        rank_for_p3_by_matrix[name] = rank
                        print("Rank @p3 for %s : %i" % (name, rank))

                    if (
                        (classificationTask and valid_acc >= perf_thresholds[1] or not classificationTask and valid_loss <= perf_thresholds[1])
                        and name not in rank_for_p2_by_matrix
                    ):
                        rank_for_p2_by_matrix[name] = rank
                        print("Rank @p2 for %s : %i" % (name, rank))

                    if (
                        (classificationTask and valid_acc >= perf_thresholds[0] or not classificationTask and valid_loss <= perf_thresholds[0])
                        and name not in rank_for_p1_by_matrix
                    ):
                        rank_for_p1_by_matrix[name] = rank
                        print("Rank @p1 for %s : %i" % (name, rank))
                        go = False  # skip the rest of the ranks
                else:
                    svd_error_by_rank.append(np.nan)
                    loss_by_rank.append(np.nan)
                    accuracy_by_rank.append(np.nan)

            svd_error_by_rank_by_matrix[name] = svd_error_by_rank
            loss_by_rank_by_matrix[name] = loss_by_rank
            accuracy_by_rank_by_matrix[name] = accuracy_by_rank

    # create figures
    NUM_COLORS = 12
    LINE_STYLES = ["solid", "dashed", "dashdot", "dotted"]
    NUM_STYLES = len(LINE_STYLES)
    cm = plt.get_cmap("gist_rainbow")

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 30), sharex=False)

    for i, (name, svd_error_by_rank) in enumerate(svd_error_by_rank_by_matrix.items()):
        accuracy_by_rank = accuracy_by_rank_by_matrix[name]
        loss_by_rank = loss_by_rank_by_matrix[name]

        lines1 = ax1.semilogy(ranks, svd_error_by_rank, label=name)
        lines2 = ax2.plot(ranks, accuracy_by_rank, label=name)
        lines3 = ax3.plot(ranks, loss_by_rank, label=name)

        for lines in (lines1, lines2, lines3):
            lines[0].set_color(cm(i / NUM_COLORS))
            lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

    for ax in (ax1, ax2, ax3):
        # Shrink current axis by 25%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

        ax.grid()
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Rank")

    ax1.set_ylabel("SVD error")
    ax2.set_ylabel("Accuracy")
    ax3.set_ylabel("Loss")

    plt.savefig("figures/%s/%s/choose_rank_for_each_matrix.png" % (task_name, modelname))

    return (rank_for_p1_by_matrix, rank_for_p2_by_matrix, rank_for_p3_by_matrix)


def evaluate_rank_tuning(state, rank_per_matrix, save_to_file=False, path_to_save=""):
    """
    This function takes a model and a dictionnary of ranks, and outputs the score
    of the model when each matrix is factorised using the rank indicated in the dictionnary
    """

    modules = {}
    memory = 0
    affected_matrices = 0

    # used to save the low-rank version of the model
    state_with_factorised_matrices = {}

    for name, tensor in state.items():
        if name in rank_per_matrix:
            rank = rank_per_matrix[name]

            U, S, V = torch.linalg.svd(tensor, full_matrices=False)
            Ur, Sr, Vr = U[:, :rank], S[:rank], V[:rank, :]

            newtensor = torch.matmul(Ur * Sr, Vr)
            error = torch.linalg.norm(newtensor - tensor).item()
            print(
                "For matrix %s, the LRA error is: %0.1f, rank used is %i"
                % (name, error, rank)
            )

            modules[name] = newtensor.to(device)
            memory += (
                Ur.numel() * 4 + Vr.numel() * 4
            )  # ignoring Sr because it can be stored with Ur
            affected_matrices += 1

            if save_to_file:
                # it is necessary to clone tensors otherwise they will share space with the bigger matrices
                state_with_factorised_matrices[name + "_left"] = (Ur * Sr).clone()
                state_with_factorised_matrices[name + "_right"] = Vr.clone()
        else:
            modules[name] = tensor.to(device)
            memory += tensor.numel() * 4

            if save_to_file:
                # fix for rank-1 matrices
                if "weight_ih" in name:
                    state_with_factorised_matrices[name + "_left"] = tensor.clone()
                    state_with_factorised_matrices[name + "_right"] = torch.ones(1, 1)
                else:
                    state_with_factorised_matrices[name] = tensor.clone()

    # do a forward pass over the dataset
    valid_acc, valid_loss = valid_accuracy(model, modules, state)
    print("Memory used: %i, affected matrices: %i" % (memory, affected_matrices))

    if save_to_file:
        torch.save(state_with_factorised_matrices, path_to_save)

    return valid_loss, valid_acc, memory, affected_matrices


# -------------------------------------------------------------------------

device = "cuda:0"
train_size = 0.8
seed = 91741
bz = 256  # batch size
task_name = "SequentialMNIST"

if task_name == "DOCC10":
    from GoGRU import GoGRU
    from DOCC10 import DOCC10, load_DOCC10_data

    classificationTask = True
    dataset_path = str(Path.home()) + "/datasets/DOCC10/DOCC10_train/"

    models = {
        "Baseline": "models/DOCC10_noLRA/gogru.pt",
        "NR+HLRA": "models/DOCC10_LRA/gogru.pt",
    }

    _, _, X_val_std, y_val = load_DOCC10_data(dataset_path, train_size, seed, load_from_pickle=True)
    valset = DOCC10(X_val_std, y_val)

    modelConstructor = GoGRU
    criterion = torch.nn.CrossEntropyLoss()

    # ranks and thresholds to be used in the experiments
    ranks = list(range(10, 300 + 1, 10))
    ranks_for_ranktuning = np.arange(1, 50 + 1, 1)
    def threshold_maker(name): return 15 if name == "Baseline" else 2
    def perf_thresholds_maker(name): return (0.916, 0.914, 0.911) if name == "Baseline" else (0.910, 0.908, 0.905)
elif task_name == "SequentialMNIST":
    from GoGRU import GoGRU_sequence
    from SequentialMNIST import SequentialMNIST

    classificationTask = False

    models = {
        "Baseline": "models/SequentialMNIST/gogru_SequentialMNIST_21072022-132652.pt",
        "NR+HLRA": "models/SequentialMNIST/gogru_SequentialMNIST_21072022-120843.pt",
    }

    dataset_test = MNIST(
        root=str(Path.home()) + "/datasets", train=False, download=True
    )
    valset = SequentialMNIST(dataset_test)

    modelConstructor = GoGRU_sequence
    criterion = torch.nn.MSELoss()

    # ranks to be considered
    ranks = list(range(10, 200 + 1, 10))
    ranks_for_ranktuning = np.arange(1, 120 + 1, 1)
    def threshold_maker(name): return 20 if name == "Baseline" else 10
    def perf_thresholds_maker(name): return (0.309, 0.311, 0.315) if name == "Baseline" else (0.309, 0.311, 0.315)
else:
    print("Unknown task")
    exit(1)

print("Number of samples in validation set: %i" % len(valset))
valloader = DataLoader(valset, batch_size=bz, shuffle=True, drop_last=True)

# load model(s)
for name, file_name in models.items():
    model = modelConstructor()

    state = torch.load(file_name)
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    models[name] = (model, state)

# print model(s) information
for name, (model, state) in models.items():
    basemodel_params = sum(p.numel() for p in model.parameters())

    print("Model %s" % name)
    print("Number of parameters: %i\n" % basemodel_params)

    rnn_weights_params = 0
    for name, tensor in state.items():
        if "gru.weight" in name:
            print(name, tensor.shape)
            rnn_weights_params += tensor.shape[0] * tensor.shape[1]

    print(
        "\nRNN weights represent %0.1f%% of the parameters"
        % (rnn_weights_params / basemodel_params * 100)
    )
    print("----------\n")

# analyse singular values for each matrix of each model
# create figures
for modelname, (model, state) in models.items():
    print("Model %s" % modelname)

    f, ax = plt.subplots(1, 1, figsize=(15, 15))

    sum_of_fro_norm = 0
    sum_of_nuc_norm = 0

    c = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for name, tensor in state.items():
        if "gru.weight" in name and min(tensor.shape) > 1:
            W = tensor.cpu().numpy()
            U, S, V = np.linalg.svd(W, full_matrices=False)

            sum_of_nuc_norm += np.sum(S)
            sum_of_fro_norm += np.linalg.norm(W)

            ax.semilogy(
                S,
                "--" if "hh" in name else "-",
                label=name,
                color=c[int(name.split("_")[2][1])],
            )

            print("%34s %6.1f" % (name, np.sum(S)), W.shape)

    plt.xlabel("Index")
    plt.title("[Model=%s] Singular values by decreasing order" % modelname)
    plt.legend(loc="best")
    os.makedirs("figures/%s/%s/" % (task_name, modelname), exist_ok=True)
    plt.savefig("figures/%s/%s/singular_values.png" % (task_name, modelname))

    print("Sum of nuclear norms: %0.1f" % sum_of_nuc_norm)
    print("Sum of Frobenius norms: %0.1f" % sum_of_fro_norm)
    print("----------\n")

# compute accuracy of model without any rank modification
for name, (model, state) in models.items():
    print("Base accuracy of model %s" % name)
    _ = valid_accuracy(model, state)
    print("----------\n")

results_dict = {}

# for each model, try several strategies of rank assignation
for name, (model, state) in models.items():
    print("Evaluating model %s" % name)

    print("Strategy #1 : use same rank for all matrices")

    xp_name = "%s" % (name)
    results_dict[xp_name] = factorize_using_same_rank_for_all_matrices(
        state, ranks, 10**10
    )
    print("----------\n")

    print("Strategy #2 : only consider 'ih' matrices")

    xp_name = "%s - ih only" % (name)
    results_dict[xp_name] = factorize_using_same_rank_for_all_matrices(
        state, ranks, 10**10, ih_only=True
    )
    print("----------\n")

    print(
        "Strategy #3 : use an error threshold to decide if a matrix will be factorised or not"
    )

    threshold = threshold_maker(name)
    xp_name = "%s - Adaptative (%i)" % (name, threshold)

    results_dict[xp_name] = factorize_using_same_rank_for_all_matrices(
        state, ranks, threshold
    )
    print("----------\n")

    print("Strategy #4 : rank-tuning. Searching for optimal ranks...")

    xp_name = "%s - Rank-tuning" % (name)

    perf_thresholds = perf_thresholds_maker(name)
    p1, p2, p3 = find_optimal_rank_for_each_matrix(state, ranks_for_ranktuning, perf_thresholds, name)

    print("Evaluating model with optimal ranks...")

    losses = []
    accs = []
    memories = []
    affected_matrices = []
    for i, p_tab in enumerate((p1, p2, p3)):
        print("@ precision p%i" % (i + 1))
        save_to_file = False

        if i == 0:
            save_to_file = True
            path_to_save = "models/%s/gogru_%s_factorised.pt" % (
                task_name, "noLRA" if name == "Baseline" else "LRA"
            )

        loss, acc, memory, matrices = evaluate_rank_tuning(
            state, p_tab, save_to_file, path_to_save
        )
        losses.append(loss)
        accs.append(acc)
        memories.append(memory)
        affected_matrices.append(matrices)

    results_dict[xp_name] = (losses, accs, memories, affected_matrices)

    print("----------\n")

# saves the dictionnary of results to a file
os.makedirs("pickle/results_dicts/", exist_ok=True)
with open("pickle/results_dicts/%s.pkl"%task_name, "wb") as f:
    pickle.dump(results_dict, f)

# produces final figure with memory-precision trade-off

fig = plt.figure(figsize=(15, 15))

baseline_size = results_dict["Baseline"][2][-1]
baseline_perf = results_dict["Baseline"][1][-1] if classificationTask else results_dict["Baseline"][0][-1]

plt.xlabel("% of baseline size")
plt.ylabel("% of baseline performance")

colors = [
    "red",
    "brown",
    "salmon",
    "darkorange",
    "darkolivegreen",
    "green",
    "lime",
    "palegreen",
    "navy",
    "blue",
    "deepskyblue",
    "aqua",
]

for i, (xp_name, (loss_br, acc_br, memory_br, _)) in enumerate(results_dict.items()):
    color = colors[i]
    marker = "o"
    perf_br = acc_br is classificationTask else loss_br

    plt.scatter(
        np.array(memory_br) / baseline_size * 100,
        np.array(perf_br) / baseline_perf * 100,
        label=xp_name,
        color=color,
        marker=marker,
        s=70,
    )

plt.axvline(100, label="Baseline memory cost", ls="--", color="black")
plt.axhline(100, label="Baseline performance", ls="--", color="black")
# plt.axvline(28254*4/baseline_size*100, label="Minimal size for this architecture", ls='--', color='red')

if classificationTask:
    plt.ylim((90, 102))
else:
    plt.ylim((98, 110))

plt.grid()
plt.legend(loc="best")
plt.title("Size vs accuracy trade-off")
plt.savefig("figures/%s/size_accuracy_trade_off.png"%task_name)
