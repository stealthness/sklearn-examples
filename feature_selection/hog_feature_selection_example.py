"""
The purpose of this file is to explore the HOG in sklearn
"""
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from skimage.feature import hog

from sw_path import WORK_ROOT

# load a simple image of a car

file = WORK_ROOT+'RES\TestFolder\Cars\car1.jpg'

img = image.imread(file)

# create a figure with 2 subplots
fig, (sub1, sub2) = plt.subplots(1, 2)

# add the original image to subplot 1
sub1.imshow(img)
sub1.axis('off')
sub1.set_title('Original Image of a Red Car')

# create HoG
fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

# add the original image to subplot 1
sub2.imshow(hog_img)
sub2.axis('off')
sub2.set_title('HoG of a Red Car')

# show figure
fig.suptitle('Histogram of Gradients Example')
fig.show()

