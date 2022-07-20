# coding: utf-8

"""
Date: 28-06-2022

Author: Lucas Maison

Contains the definition of the GoGRU model

GoGRU_sequence is a variant of GoGRU useful for sequence-to-sequence tasks
"""

import torch


class GoGRU(torch.nn.Module):
    def __init__(self, num_layers=3, hidden_size=150, dropout=0.2, bidirectional=True):

        super(GoGRU, self).__init__()

        bidir_factor = 2 if bidirectional else 1

        self.gru = torch.nn.GRU(
            input_size=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.fc_out = torch.nn.Linear(bidir_factor * hidden_size, 10)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x, _ = x.max(dim=1)
        x = self.fc_out(x)
        # x = self.dropout(x)

        return x


class GoGRU_sequence(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_size=100, dropout=0.2, bidirectional=True):

        super(GoGRU_sequence, self).__init__()

        bidir_factor = 2 if bidirectional else 1

        self.gru = torch.nn.GRU(
            input_size=1,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.fc_out = torch.nn.Linear(bidir_factor * hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = torch.relu(x)
        x = self.fc_out(x)

        return x
