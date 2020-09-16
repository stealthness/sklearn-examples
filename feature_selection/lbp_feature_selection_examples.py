"""
The purpos of this file is example code of using lbp feature selection

lbp is Local Binaray Pattern

see wiki article : https://en.wikipedia.org/wiki/Local_binary_patterns
sklean : https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
"""
import skimage
import numpy as np
from matplotlib import image
from sw_path import WORK_ROOT
from skimage.feature import local_binary_pattern

# settings for LBP
radius = 4
n_points = 8 * radius


def main():

    # load a simple image of a car

    file = WORK_ROOT+'RES\TestFolder\Cars\car1.jpg'
    color_img = image.imread(file)

    # details of the image
    print_car_details('Red Car', color_img)

    # Cannot apply lbp to a color image
    # Applying lbp to Black and White img
    bw_img = skimage.color.rgb2gray(color_img);
    bw_fd = local_binary_pattern(bw_img, n_points, radius, method='uniform')

    print_car_details("BW Red Car", bw_img)

    # properties of fd
    print(f'the feature vector is of type {bw_fd.dtype}')
    print(f'the feature vector has the min/max values {np.min(bw_fd):.4f}/{np.max(bw_fd):.4f}')


def print_car_details(title: str, img: np.ndarray):
    """
    Prints the details of an image
    :param title: Title of the image
    :param img: an image in an np.ndarray
    """
    print(f'the image {title} is of type {img.dtype}')
    print(f'the image {title} has the min/max values {np.min(img):.4f}/{np.max(img):.4f}')
    print(f'The image {title} has the shape {img.shape}')
    print(f'the image {title} has size {img.size}\n')


main()