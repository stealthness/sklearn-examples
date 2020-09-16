'''
Trying to creating a python package of useful function
'''
import csv
import random
import matplotlib.pyplot as plt
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


def imshow_mean_img(img, shape, title):
    plt.imshow(img.reshape(shape), cmap='gray')
    plt.title(title)
    plt.show()


def select_random_target(b):
    choice = random.randint(0, 400)
    return b.data[choice], b.target[choice]


def show_class_images(b, choice):
    print(f'choice is {choice}  {b.target}')
    index = b.target.index(choice)
    print(f'index: {index} url:{b.data_urls[index]}')
    fig = plt.figure()
    for i in range(10):
        img =b.data[index+i].reshape(b.shape)
        fig.add_subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    fig.suptitle(f'Class {b.target[index]} url:{b.data_urls[index]}')
    plt.show()
