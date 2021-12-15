#!/opt/homebrew/bin/python3
import numpy as np
from PIL import Image
import glob

def realH(img_size):
    images = glob.glob("/home/emma/Documents/images/*.png")
    hold = np.zeros((20, img_size, img_size))
    i = 0
    for image in images:
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.convert('L')
            img = img.resize((img_size, img_size), resample=Image.LANCZOS)
            img = np.asarray(img)
            img = (img / 255) - 0.5
            hold[i] = img
            i = i + 1

    dims = np.load("/home/emma/src/lndpd/realdata_label.npy")

    return hold, dims
