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
    ret_val, image = cam.read()
    #cv.imwrite("image.png", image)
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LANCZOS4)
    image = cv.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    image = cv.normalize(image, None, alpha=-0.5, beta=0.5, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    #np.save('image.npy', image)
    image = torch.from_numpy(image).to(device)
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

# define mipi parms
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=120,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if __name__ == '__main__':
    ### define data paths and vars###
    img_size = 48
    PATH = "state_dict_model.pt"
    iteration = 100

    ### call model ###
    model = get_model()

    ### load model ###
    model.load_state_dict(torch.load(PATH))
    model.eval()

    ### define cam ###
    cam = cv.VideoCapture(gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)
    if not cam.isOpened():
        print("Failed to open camera")

    start = time.time()
    for t in range(iteration):
        X = get_image()
        y = infer(X)
        print(f'ind: {t+1} -- pred: {y[0]*img_size}')
    end = time.time()

    print((end - start) / iteration)

    cam.release()
