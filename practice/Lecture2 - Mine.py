# %% Import necessary libraries
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

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


# %% TRAINING
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 150

# Loop over the number of epochs
for i in range(epochs):

    # Loop over all sequences in the training set
    for seq, labels in train_inout_seq:

        # Zero out the gradients from the optimizer. This is done because
        # gradients are accumulated in PyTorch, so they need to be reset each time.
        optimizer.zero_grad()

        # Reset the hidden state. The model uses this to store information
        # from previous time steps, but we want to start fresh for each sequence.
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        # Run the sequence through the model to get a prediction
        y_pred = model(seq)

        # Calculate the loss between the prediction and the true value
        single_loss = loss_function(y_pred, labels)

        # Backpropagate the error, this calculates the gradients for all model parameters
        single_loss.backward()

        # Step the optimizer, this updates the model parameters based on the gradients
        optimizer.step()

    # Print the loss every 25 epochs
    if i % 25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # Print the final loss
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


# %% GENERATE PREDICTIONS
fut_pred = 12

# Initialize a list with the last 'train_window' data points from the normalized dataset
# These will be used as starting point for the predictions
test_inputs = data_normalized[-train_window:].tolist()

# Set the model to evaluation mode
model.eval()

# Loop over the range of future predictions (fut_pred)
for i in range(fut_pred):
    
    # Prepare the sequence to feed into the model, it contains the last 'train_window' data points
    seq = torch.FloatTensor(test_inputs[-train_window:])
    
    # torch.no_grad() impacts the autograd engine and deactivate it. It will reduce memory usage and
    # speed up computations but we won’t be able to perform backpropagation.
    # We don’t need gradients for validation/testing.
    with torch.no_grad():
        # Reset the hidden state. The model uses this to store information
        # from previous time steps, but we want to start fresh for each prediction.
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # Append the prediction to the list of test_inputs.
        # The model(seq) is producing the prediction.
        test_inputs.append(model(seq).item())

# Transform the normalized predictions to their original scale by applying the inverse transformation
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print('Finished prediction.')


#%% FINAL ERROR

# Actual and predicted values
actual = scaler.inverse_transform(np.array(data_normalized[train_window:] ).reshape(-1, 1))
predicted = actual_predictions
# Calculate MAE and RMSE
mae = mean_absolute_error(actual, predicted)
rmse = math.sqrt(mean_squared_error(actual, predicted))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
69/106 RNNs & Sequence Data Processing - Applications & Architectures of Deep Learning - MMA - Desautels Faculty of Management, McGill
