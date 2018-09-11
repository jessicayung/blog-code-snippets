import torch
import torch.nn as nn
from generate_data import *
import matplotlib.pyplot as plt

# data params
noise_var = 0
num_datapoints = 100
test_size = 0.2
num_train = int((1-test_size) * num_datapoints)

# network params
input_size = 20
per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
h1 = 256
D_out = 1
learning_rate = 1e-2
num_epochs = 500
dtype = torch.float

# generate data
# data = SineWaveData(num_datapoints, num_prev=input_size, test_size=test_size, max_t=100, amplitude=0.5, mean=0.5)

data = ARData(num_datapoints, num_prev=input_size, test_size=test_size, noise_var=noise_var, coeffs=fixed_ar_coefficients[input_size])

# make training and test sets in torch
X_train = torch.from_numpy(data.X_train).type(torch.Tensor)
X_test = torch.from_numpy(data.X_test).type(torch.Tensor)
y_train = torch.from_numpy(data.y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(data.y_test).type(torch.Tensor).view(-1)

X_train = X_train.view([input_size, -1, 1])
X_test = X_test.view([input_size, -1, 1])

# build model

class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        print("Batch size: ", batch_size)

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)  # 1 here is output size

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = RNN(1, 32, batch_size=num_train)
"""
model = torch.nn.Sequential(
    torch.nn.LSTM(input_size=lstm_input_size, hidden_size=h1, num_layers=1, dropout=0),
    torch.nn.ReLU(),
    torch.nn.Linear(h1, out_features=D_out)
)
"""

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)
# train model
for t in range(num_epochs):
    model.zero_grad()
    model.hidden = model.init_hidden()
    # forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    optimiser.step()

plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()