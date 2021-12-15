import numpy as np

def artificial(img_size):
    num_imgs = 5000
    rng = np.random.default_rng()

    img = rng.random((num_imgs, img_size,img_size), dtype="float32")

    dims = np.zeros((num_imgs, 4), dtype="float32")

    for z in range(num_imgs):
        x_scale = int(np.round(rng.random() + 1))
        y_scale = int(np.round(rng.random() + 1))
        h_width = 10*x_scale
        h_height = 13*y_scale

        temp = np.zeros((h_height,h_width), dtype="float32")
        temp[0:5*y_scale, 3*x_scale:7*x_scale] = (0.55 - 0.45)*rng.random() + 0.45
        temp[8*y_scale:13*y_scale, 3*x_scale:7*x_scale] = (0.55 - 0.45)*rng.random() + 0.45

        x = np.random.randint(0, img_size - h_width)
        y = np.random.randint(0, img_size - h_height)
        for i in range(h_width):
            for j in range(h_height):
                img[z, y+j, x+i] = temp[j, i]

        if int(np.round(rng.random())) < 0.5:
            img[z] = img[z].T
            dims[z] = np.divide([y+(h_height/2), x+(h_width/2), h_height, h_width], img_size)
        else:
            dims[z] = np.divide([x+(h_width/2), y+(h_height/2), h_width, h_height], img_size)

    img = img - 0.5

    return img, dims
