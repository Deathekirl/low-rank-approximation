#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 15-07-2022

Author: Lucas Maison

Visualize training metrics (.npy files) by producing a figure
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 12))

name = "DOCC10_29062022-122330_%s.%s"
path = "npy/" + name % ("%s", "npy")
train_losses = np.load(path % "train_losses")
val_losses = np.load(path % "val_losses")
train_accs = np.load(path % "train_accs")
val_accs = np.load(path % "val_accs")
trace_norms = np.load(path % "trace_norms")

ax1.plot(train_losses)
ax2.plot(val_losses)
ax3.plot(trace_norms)
ax4.plot(train_accs)
ax5.plot(val_accs)

ax1.set_ylabel("Train loss")
ax2.set_ylabel("Val loss")
ax3.set_ylabel("Trace norm")
ax4.set_ylabel("Train acc")
ax5.set_ylabel("Val acc")

f.delaxes(ax6)

for ax in (ax1, ax2, ax3, ax4, ax5):
    ax.grid()
    ax.set_xlabel("Epochs")

f.tight_layout()

os.makedirs("figures/", exist_ok=True)
plt.savefig("figures/" + name % ("metrics", "png"))
