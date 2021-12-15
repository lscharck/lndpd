#!/usr/bin/python3.8
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor, Lambda, Compose

# check for cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

bs = 1
epochs = 100
#lr = 0.001
#momentum = 0.9

x_train = torch.from_numpy(np.expand_dims(np.load("image.npy"), (-1, -2))).T.float()
y_train = torch.from_numpy(np.expand_dims(np.load("label.npy"), -1)).T.float()

#x_train = np.expand_dims(x_train, axis=-1)
x_valid = x_train
y_valid = y_train
print(x_train.size(), y_train.size())
#x_train = torch.rand(1000, 1, 28, 28, device=device)
#y_train = torch.rand(1000, 4, device=device)
#x_valid = torch.rand(500, 1, 28, 28, device=device)
#y_valid = torch.rand(500, 4, device=device)

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

# Define CNN arch
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.AvgPool2d((2,2))
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*13*13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        return logits

# Define loss function
loss_func = F.mse_loss

# Batch loss and updater/validater function
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# Define training function
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            mse, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        avg_mse = np.sum(mse) / np.sum(nums)

        print(f'Epoch: {epoch+1}/{epochs}, avg_mse = {avg_mse}')

# Setup model
def get_model():
    model = CNN().to(device)
#    return model, optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return model, optim.Adam(model.parameters())

# Get and format data
def get_data(train_ds, valid_ds, bs):
    return (
    DataLoader(train_ds, batch_size=bs, shuffle=True),
    DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
