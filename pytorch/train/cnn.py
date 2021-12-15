#!/usr/bin/python3.8
import time
import loss
import arch
import torch
import numpy as np
from torch import nn
from torch import optim
from dataset import LandingPadH, ToTensor
from torch.utils.data import DataLoader

# check for cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
threads = torch.get_num_threads()
print(f'Using {device} device\n{threads} cpu threads')

# Define loss function
loss_func = loss.CIoULoss()

# Batch loss and updater/validater function
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), loss_func.avg_iou.item(), len(xb)

# Define training function
def fit(epochs, model, loss_func, opt, sched, train_dl, valid_dl):
    for epoch in range(epochs):
        start = time.time()
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        end = time.time()

        model.eval()
        with torch.no_grad():
            CIoU, iou, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        avg_CIoU = np.sum(np.multiply(CIoU, nums)) / np.sum(nums)
        avg_iou = np.sum(np.multiply(iou, nums)) / np.sum(nums)
        sched.step(avg_CIoU)

        print(f'Epoch: [{epoch+1}/{epochs}]\t({end-start:.2g}s)\t-- avg_loss = {avg_CIoU:.2E}\t-- avg_acc = {avg_iou:.2g}')

# Setup model
def get_model():
    model = arch.CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=0.001, verbose=True)
    return model, optimizer, scheduler

# Get and format data
def get_data(train_ds, valid_ds, bs):
    return (
    DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2),
    DataLoader(valid_ds, batch_size=bs * 2, num_workers=2),
    )

if __name__ == '__main__':
    ### define data paths and vars###
    bs = 32
    epochs = 1000
    img_size = 32
    lr = 0.001
    train_label_path = "/home/emma/Documents/train_labels.npy"
    valid_label_path = "/home/emma/Documents/valid_labels.npy"
    train_data_path = "/home/emma/Documents/data/train"
    valid_data_path = "/home/emma/Documents/data/valid"

    ### create data sets ###
    train_ds = LandingPadH(label_file=train_label_path, root_dir=train_data_path, transform=ToTensor())
    valid_ds = LandingPadH(label_file=valid_label_path, root_dir=valid_data_path, transform=ToTensor())

    print(f'Train dataset size: {len(train_ds)}\nValid dataset size {len(valid_ds)}')

    ### call and train model ###
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt, sched = get_model()
    fit(epochs, model, loss_func, opt, sched, train_dl, valid_dl)

    ### save model ###
    PATH = "../inference/state_dict_model.pt"
    torch.save(model.state_dict(), PATH)


    ### evaluation ###
    model.eval()
    with torch.no_grad():
        for xb, yb in valid_dl:
            loss = loss_func(pred := model(xb), yb)
            print(f'loss = {loss:.2E} -- iou = {loss_func.avg_iou:.2g}')
            print(pred[0]*img_size, yb[0]*img_size)
            print(pred[1]*img_size, yb[1]*img_size)
            print(pred[2]*img_size, yb[2]*img_size)
            print(pred[3]*img_size, yb[3]*img_size)
            print(pred[4]*img_size, yb[4]*img_size)
            break
