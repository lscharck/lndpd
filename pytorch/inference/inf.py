#!/usr/bin/python3
import time
import arch
import torch
import cv2 as cv
import numpy as np

# check for cpu or gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
threads = torch.get_num_threads()
print(f'Using {device} device\n{threads} cpu threads')

# image capture
def get_image():
    status, image = webcam.read()
    #cv.imwrite("image.png", image_in)
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LANCZOS4)
    image = cv.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    image = cv.normalize(image, None, alpha=-0.5, beta=0.5, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    #np.save('image.npy', image)
    image = torch.from_numpy(image)
    image = image.expand(1, 1, img_size, img_size)

    return image

# infer
def infer(image):
    with torch.no_grad():
        pred = model(image)

    return pred

# Setup model
def get_model():
    model = arch.CNN().to(device)

    return model

if __name__ == '__main__':
    ### define data paths and vars###
    img_size = 48
    PATH = "state_dict_model.pt"
    iteration = 200

    ### call model ###
    model = get_model()

    ### load model ###
    model.load_state_dict(torch.load(PATH))
    model.eval()

    ### define cam ###
    webcam = cv.VideoCapture(0)
    webcam.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 1280)

    start = time.time()
    for t in range(iteration):
        X = get_image()
        y = infer(X)
        print(f'ind: {t+1} -- pred: {y[0]*img_size}')
    end = time.time()

    print((end - start) / iteration)

    webcam.release()
