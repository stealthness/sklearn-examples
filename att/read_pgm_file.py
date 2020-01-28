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
import numpy as np
import matplotlib.pyplot as plt


def read_pgm(name):
    with open(name, 'rb') as f:
        d = []
        if f.readline() == b'P5\n':
            b = f.readline()
            s = b.decode('utf-8').strip().split(" ")
            shape = (int(s[1]), int(s[0].strip()))
            max_value = int(f.readline().strip())
            for byte in f.readline():
                d.append(byte)
        else:
            print('P5 not found')
            return None

    return shape, max_value, np.array(d)


sh, mv,  img = read_pgm('D:\\RES\\ORL\\s1\\1.pgm')

plt.imshow(img.reshape(sh), cmap='gray')
plt.show()
