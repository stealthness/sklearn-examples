""""
The purpose of this file is to show code to retrieve images

"""
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from sw_path import WORK_ROOT


def get_file_name(file):
    return (file.split('//')[-1]).split('.')[0]


def display_img(file_input):
    file = file_input.replace('\\', '//')
    img = image.imread(file);
    name = get_file_name(file)
    print(f'the image {name} is of type {img.dtype}')
    print(f'the image {name} has the min/max values {np.min(img):.4f}/{np.max(img):.4f}')
    print(f'The image {name} has the shape {img.shape}')
    plt.imshow(img)
    plt.title(name)
    plt.axis("off")
    plt.show()

# Load an image from this folder
display_img('lenna.png')

# lets load an image from a different
display_img(WORK_ROOT+'RES\TestFolder\Cars\car1.jpg')