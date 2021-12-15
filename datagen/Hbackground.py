#!/opt/homebrew/bin/python3
import numpy as np
from PIL import Image
import os

def backgrounds(img_size):
    num_imgs = 120
    rotations = 4
    perm = 8
    num_imgs_total = num_imgs * rotations * perm
    dims = np.zeros((num_imgs_total, 4), dtype="float32")
    hold = np.zeros((num_imgs_total, img_size, img_size))
    y = 0
    for i in range(num_imgs):
        with open(os.path.join('/home','emma','Documents','backgrounds', 'image' + str(i+1) + '.png'), 'rb') as file:
            img = Image.open(file)
            img = img.resize((img_size, img_size), resample=Image.LANCZOS)
            img = img.convert('RGB')
            img = img.convert('L')
            for r in range(rotations):
                for p in range(perm):
                    fillH(img_size, hold, dims, y, img = img.rotate(r * 90, expand=1))
                    y+=1

    return hold, dims

def fillH(img_size, hold, dims, i, img):
    rng = np.random.default_rng()
    img = np.array(img)

    x_scale = int(np.round(rng.random() + 1))
    y_scale = int(np.round(rng.random() + 1))
    h_width = 10*x_scale
    h_height = 13*y_scale

    temp = np.ones((h_height,h_width), dtype="float32") * ((110 - 90) * rng.random() + 90)
    temp[0:5*y_scale, 3*x_scale:7*x_scale] = (210 - 190) * rng.random() + 190
    temp[8*y_scale:13*y_scale, 3*x_scale:7*x_scale] = (210 - 190) * rng.random() + 190

    x = np.random.randint(0, img_size - h_width)
    y = np.random.randint(0, img_size - h_height)
    for ii in range(h_width):
        for jj in range(h_height):
            img[y+jj, x+ii] = temp[jj, ii]

    if int(10 * rng.random()) > 5:
        dims[i] = np.divide([(x + (h_width / 2)), (y + (h_height / 2)), h_width, h_height], img_size)
    else:
        img = img.T
        dims[i] = np.divide([(y + (h_height / 2)), (x + (h_width / 2)), h_height, h_width], img_size)

    img = (img / 255) - 0.5
    hold[i] = img
