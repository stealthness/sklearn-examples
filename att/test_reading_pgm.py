import errno
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


def readpgm(name):
    with open(name) as f:
        zero = f.read(1);
        print(zero)
        #lines = f.readlines()

    # This ignores commented lines
    return "beast"

#data = readpgm('I:/RES/ATT/s1/1.pgm')

#plt.imshow(np.reshape(data[0],data[1])) # Usage example


def read_images(path, sz=None):
    c = 0
    X = []
    y = []
    for dirname , dirnames , filenames in os.walk ( path ):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                print(filename)
                print(subject_path)
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert ("L")
                    # resize to given size (if given )
                    if sz is not None:
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError :
                    print("I/O error ({0}) : {1} ". format(errno, os.strerror))
                except :
                    print("Unexpected error :", sys.exc_info()[0])
                raise
        c = c+1
    return [X, y]


data = read_images('I:/RES/ATT')
print(data)
print(data.size)


