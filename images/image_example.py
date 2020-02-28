""""
The purpose of this file is to show code to retrieve images

"""
import numpy as np
import skimage.color
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sw_path import WORK_ROOT


def get_file_name(file):
    return (file.split('//')[-1]).split('.')[0]


def display_plot(plt_img, plt_name, **kwargs):
    if 'cmap' in kwargs:
        plt.imshow(plt_img, cmap=kwargs['cmap'])
    else:
        plt.imshow(plt_img)
    plt.title(plt_name)
    plt.axis("off")
    plt.show()


def display_img(file_input):
    file = file_input.replace('\\', '//')
    img = image.imread(file);
    name = get_file_name(file)
    print(f'the image {name} is of type {img.dtype}')
    print(f'the image {name} has the min/max values {np.min(img):.4f}/{np.max(img):.4f}')
    print(f'The image {name} has the shape {img.shape}')
    display_plot(img,name)


# Load an image from this folder
display_img('lenna.png')

# lets load an image from a different
display_img(WORK_ROOT+'RES\TestFolder\Cars\car1.jpg')

# converting to greyscale
c_img = image.imread('lenna.png')
bw_img = skimage.color.rgb2gray(c_img);
display_plot(bw_img,'bw car', cmap='gray')


# messing about with scalers
fig, (sub1, sub2)= plt.subplots(1,2)

hist = np.hstack(bw_img.flatten())
sub1.hist(hist)
sub1.set_title('Original data unscaled')


scaler = StandardScaler()
# th
bw_img = scaler.fit_transform(bw_img)
print(f'the image BW Scaled Lenna is of type {bw_img.dtype}')

hist = np.hstack(bw_img.flatten())
sub2.hist(hist)
sub2.set_title('Scaled in Guassian distribution')
plt.show()

# here we can see that the