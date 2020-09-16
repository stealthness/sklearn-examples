import errno
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import Bunch

from orl_face_dataset_examples.read_pgm_file import read_pgm, fetch_sw_orl, plot_img
from sw_path import WORK_ROOT
from sw_utils.functions import show_class_images

control = [False, False, True]


ORL_PATH = os.path.join(WORK_ROOT, 'RES', 'ORL')

b = fetch_sw_orl()

if control[0]:
    print(f'the data set is {b.DESCR}')
    print(f'the data is located at {b.path}')

    data_shape = b.shape
    data_size = b.data.size
    print(f'The shape of the data is {data_shape}, that is there are {data_size} number of images')

if control[1]:
    img = np.array(b.data[random.randint(0, 400 - 1)].reshape(data_shape))
    plt.imshow(img, cmap='gray')
    plt.show()
    fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                      multichannel=False)
    plt.imshow(hog_img)
    plt.show()

    hog_fd = []
    for img in b.data:
        if data_size == len(img):
            fd = hog(np.array(img).reshape(data_shape), orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                     visualize=False, multichannel=False)


if control[2]:
    show_class_images(b, 's33')
