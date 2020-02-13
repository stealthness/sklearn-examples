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

# load a simple image of a car

file = WORK_ROOT+'RES\TestFolder\Cars\car1.jpg'
img = image.imread(file)

# details of the image
name = 'Red Car'
print(f'the image {name} is of type {img.dtype}')
print(f'the image {name} has the min/max values {np.min(img):.4f}/{np.max(img):.4f}')
print(f'The image {name} has the shape {img.shape}')
print(f'the image {name} has size {img.size}')


# Applying the lbp to a color image
#fd = local_binary_pattern(img, n_points, radius, method='uniform')

# Applying to Black and White img

bw_img = skimage.color.rgb2gray(img);
bw_fd = local_binary_pattern(bw_img, n_points, radius, method='uniform')

name = 'BW Red Car'
print(f'the image {name} is of type {bw_img.dtype}')
print(f'the image {name} has the min/max values {np.min(bw_img):.4f}/{np.max(bw_img):.4f}')
print(f'The image {name} has the shape {bw_img.shape}')
print(f'the image {name} has size {bw_img.size}')

# properties of fd
print(f'the feature vector is of type {bw_fd.dtype}')
print(f'the feature vector has the min/max values {np.min(bw_fd):.4f}/{np.max(bw_fd):.4f}')
