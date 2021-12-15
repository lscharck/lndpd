#!/usr/bin/python3
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import Hartificial
import Hbackground
import Hreal

def s2d(train_hold, valid_hold):
    for i in range(len(train_hold)):
        np.save("/home/emma/Documents/data/train/image" + str(i) + ".npy", train_hold[i])
    for i in range(len(valid_hold)):
        np.save("/home/emma/Documents/data/valid/image" + str(i) + ".npy", valid_hold[i])

def print_image(sel, hold, dims, img_size):
    plt.imshow(hold[sel], cmap='winter', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    x = (dims[sel][0] - dims[sel][2]/2) * img_size
    y = (dims[sel][1] - dims[sel][3]/2) * img_size
    plt.gca().add_patch(matplotlib.patches.Rectangle((x, y), dims[sel][2] * img_size, dims[sel][3] * img_size, ec='r', fc='none'))
    plt.show()

def assembly():
    rng = np.random.default_rng()
    img_size = 48
    hold1, dims1 = Hbackground.backgrounds(img_size)
    hold2, dims2 = Hartificial.artificial(img_size)
    hold3, dims3 = Hreal.realH(img_size)

    hold = np.concatenate((hold1, hold2))
    dims = np.concatenate((dims1, dims2))

    index = np.arange(len(hold))
    rng.shuffle(index)

    hold = hold[index]
    dims = dims[index]

    hold = np.concatenate((hold, hold3))
    dims = np.concatenate((dims, dims3))

    print(f'images: {np.shape(hold1)}\tdims: {np.shape(dims1)}')
    print(f'images: {np.shape(hold2)}\tdims: {np.shape(dims2)}')
    print(f'images: {np.shape(hold3)}\tdims: {np.shape(dims3)}')
    print(f'images: {np.shape(hold)}\tdims: {np.shape(dims)}')

    train_hold = hold[:int(len(hold) * 0.8)]
    valid_hold = hold[int(len(hold) * 0.8):]
    s2d(train_hold, valid_hold)

    train_dims = dims[:int(len(hold) * 0.8)]
    valid_dims = dims[int(len(hold) * 0.8):]
    np.save("/home/emma/Documents/train_labels.npy", train_dims)
    np.save("/home/emma/Documents/valid_labels.npy", valid_dims)

    print(f'train images: {np.shape(train_hold)}\tvalid images: {np.shape(valid_hold)}')
    print(f'train dims: {np.shape(train_dims)}\tvalid dims: {np.shape(valid_dims)}')

#    print_image(1, train_hold, train_dims, img_size)
#    print_image(200, train_hold, train_dims, img_size)
#    print_image(2000, train_hold, train_dims, img_size)


if __name__ == '__main__':
    assembly()
