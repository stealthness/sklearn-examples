'''
Trying to creating a python package of useful function
'''
import csv

import numpy as np


def import_dataset(file='I:/RES/ATT/D.txt'):
    reader = csv.reader(open(file, "r"), delimiter=",")
    x = list(reader)
    return np.array(x).astype("int")


def flatten_image(img):
    return img.reshape(92, 112)


def unflatten_image(v, dim):
    return v.resize(dim[0], dim[1])


def test_double(x):
    return 2*x