import errno
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import Bunch

from att.read_pgm_file import read_pgm

ORL_PATH = "D:\\RES\\ORL"

ATT_DESCR = '''
Our Database of Faces, (formerly 'The ORL Database of Faces'), contains a set of face images taken between April 1992 
and April 1994 at the lab. The database was used in the context of a face recognition project carried out in 
collaboration with the Speech, Vision and Robotics Group of the Cambridge University Engineering Department.

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different 
times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses 
/ no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal 
position (with tolerance for some side movement). A preview image of the Database of Faces is available.

The files are in PGM format, and can conveniently be viewed on UNIX (TM) systems using the 'xv' program. The size of 
each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each 
subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these 
directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image 
number for that subject (between 1 and 10).

The database can be retrieved from ftp://ftp.uk.research.att.com:pub/data/att_faces.tar.Z as a 4.5Mbyte compressed tar 
file or from ftp://ftp.uk.research.att.com:pub/data/att_faces.zip as a ZIP file of similar size.

A convenient reference to the work using the database is the paper Parameterisation of a stochastic model for human 
face identification. Researchers in this field may also be interested in the author's PhD thesis, Face Recognition 
Using Hidden Markov Models, available from ftp://ftp.uk.research.att.com/pub/data/fsamaria_thesis.ps.Z (~1.7 MB).

When using these images, please give credit to AT&T Laboratories Cambridge.
'''


shape, max_value, img = read_pgm('D:\\RES\\ORL\\s1\\1.pgm')


def read_images(path, sz=None):
    c = 0
    x = []
    y = []
    data = np.array([])
    target = np.array([])
    print("QQQ")
    for dir_name, dir_names, filenames in os.walk(path):
        for sub_dir_name in dir_names:
            subject_path = os.path.join(dir_name, sub_dir_name)
            print(subject_path)
            for filename in os.listdir(subject_path):
                print(filename)
                try:
                    shape, max_value, img = read_pgm(subject_path+"\\"+filename)
                    x.append(img)
                    y.append(c)
                except IOError:
                    print("I/O error ({0}) : {1} ".format(errno, os.strerror))

        c = c + 1
    b = Bunch(DESCR=ATT_DESCR, data=x, target=y, path=path)
    return b


b = read_images(ORL_PATH)
print(f'the data set is {b.DESCR}')

print(f'the data is located at {b.path}')

# data_shape = b.data.shape
# data_size = b.shape[0]
# print(f'The shape of the data is {data_shape}, that is there are {data_size} number of images')
