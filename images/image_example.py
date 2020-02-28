""""
The purpose of this file is to show code to retrieve images

"""
import numpy as np
import skimage.color
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sw_path import WORK_ROOT


def get_file_name(file: str) -> str:
    """
    Returns the filename from a give resource string
    :param file:
    :return:
    """
    return (file.split('//')[-1]).split('.')[0]


def display_plot(plt_img, plt_name, **kwargs):
    """
    Displays the plot of an image
    :param plt_img: the image to be plotted
    :param plt_name: the name of the image to be displayed in the plot
    :param kwargs:
    """
    if 'cmap' in kwargs:
        plt.imshow(plt_img, cmap=kwargs['cmap'])
    else:
        plt.imshow(plt_img)
    plt.title(plt_name)
    plt.axis("off")
    plt.show()


def display_img(image_file: str):
    """
    Displays the image from the given file location, and print out type, min/max values and shape properties of the
    the image :param image_file: the string location of a image file
    """
    file = image_file.replace('\\', '//')
    img = image.imread(file);
    name = get_file_name(file)
    print(f'the image {name} is of type {img.dtype}')
    print(f'the image {name} has the min/max values {np.min(img):.4f}/{np.max(img):.4f}')
    print(f'The image {name} has the shape {img.shape}\n')
    display_plot(img, name)


# Load an image from this folder
display_img('lenna.png')

# lets load an image from a different
display_img(WORK_ROOT+'RES\TestFolder\Cars\car1.jpg')

# converting to greyscale
c_img = image.imread('lenna.png')
bw_img = skimage.color.rgb2gray(c_img);
display_plot(bw_img, 'Black and White Car', cmap='gray')


# messing about with scalars
fig, (sub1, sub2)= plt.subplots(1,2)

hist = np.hstack(bw_img.flatten())
sub1.hist(hist)
sub1.set_title('Original Data Unscaled')


scaler = StandardScaler()
# th
bw_img = scaler.fit_transform(bw_img)
print(f'The Grayscale Image of Lenna is of type {bw_img.dtype}')

hist = np.hstack(bw_img.flatten())
sub2.hist(hist)
sub2.set_title('Scaled with a Guassian Distribution')
plt.show()
