"""
This file is to read a pgm file type
information about pgm can be found at http://netpbm.sourceforge.net/doc/pgm.html

Each PGM image consists of the following:

A "magic number" for identifying the file type. A pgm image's magic number is the two characters "P5".
Whitespace (blanks, TABs, CRs, LFs).
A width, formatted as ASCII characters in decimal.
Whitespace.
A height, again in ASCII decimal.
Whitespace.
The maximum gray value (Maxval), again in ASCII decimal. Must be less than 65536, and more than zero.
A single whitespace character (usually a newline).
A raster of Height rows, in order from top to bottom. Each row consists of Width gray values, in order from left to
right. Each gray value is a number from 0 through Maxval, with 0 being black and Maxval being white. Each gray value is
represented in pure binary by either 1 or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, it is 2
bytes. The most significant byte is first.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import Bunch

from sw_path import WORK_ROOT

RUN_SCRIPT = False
ORL_PATH = os.path.join('RES', 'ORL')


def read_pgm(filename):
    """
    Reads a pgm file and converts to a ndarry
    :param filename:
    :return: shape, max_value, ndarry
    """
    with open(filename, 'rb') as f:
        assert f.readline()[:2] == b'P5'
        s = f.readline().split()
        shape = (int(s[1]), int(s[0]))
        max_value = int(f.readline())
        assert max_value <= 255
        d = []
        for i in range(shape[0]*shape[1]):
            p = f.read(1)
            if len(p) > 0:
                d.append(ord(p))
    return shape, max_value, np.array(d)


def plot_img(img, shape):
    """
    Simple function to plot an image
    :param img:
    :param shape:
    :return:
    """
    plt.imshow(img.reshape(shape), cmap='gray')
    plt.axis('off')
    plt.title('An Example of the ORL(aka ATT) image')
    plt.show()


def read_all_pgm(dir_url):
    """
    Reads all the images in from a given folder path locations. Assumes that the folder is structured as follows
    folder_name
    |README
    |-- class1_name_folder
    |-- class2_name_folder
    |...
    |-- classN_name_folder
    :param dir_url, the location of the dataset
    :return: A bunch object, a dictionary like object with attributes
    """
    data = []
    target = []
    target_loc = []
    shape = None
    max_value = 0

    for _, dir_names, _ in os.walk(dir_url):
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dir_url, sub_dir_name)
            for filename in os.listdir(subject_path):
                try:
                    target_loc.append(os.path.join(subject_path, filename))
                    shape, max_value, img_array = read_pgm(os.path.join(subject_path, filename))
                    data.append(img_array)
                    target.append(sub_dir_name)
                except IOError:
                    print(f'I/O error : {os.strerror}')
    return shape, max_value, target, data, target_loc


def fetch_sw_orl():
    path = os.path.join(WORK_ROOT, ORL_PATH)
    with open(os.path.join(path, 'README'), 'r') as f:
        descr = f.read()
    shape, max_value, target, data, images_url = read_all_pgm(os.path.join(WORK_ROOT, ORL_PATH))
    target_list = set(target)
    return Bunch(DESCR=descr, data=np.array(data), target=target, path=path, shape=shape, max_value=max_value,
                 target_list=set(target), data_urls= images_url)


if RUN_SCRIPT:
    sh, mv, orl_name, orl_data, orl_urls = read_all_pgm(os.path.join(WORK_ROOT, ORL_PATH))
    plot_img(orl_data[56], sh)
    new_b = fetch_sw_orl()

