# %%     Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import torchvision
from torchvision import datasets, models
from torchvision import transforms

# %% DATA
flights = sns.load_dataset("flights")
data = flights["passengers"].values.astype(float)


scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))
data_normalized = torch.FloatTensor(data_normalized).view(-1)

train_window = 12


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
    train_seq = input_data[i:i+tw]
    train_label = input_data[i+tw:i+tw+1]
    inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(data_normalized, train_window)


# %% MODEL
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
