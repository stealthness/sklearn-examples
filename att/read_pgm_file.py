"""
This file is to read a pgm file type
"""
import numpy as np
import matplotlib.pyplot as plt

def read_pgm(name):
    with open(name, 'rb') as f:
        b = f.readline()
        print(b)
        d = []
        if b == b'P5\n':
            b = f.readline()
            print(b)
            s = b.decode('utf-8').strip().split(" ")
            shape = (int(s[1]), int(s[0].strip()))
            print(shape)
            max_value = int(f.readline().strip())
            print(b)

            bytes = f.readline()
            for byte in bytes:
                d.append(byte)
        else:
            print('P2 not found')
            return None

    return shape, max_value, np.array(d)


shape, max_value,  img = read_pgm('D:\\RES\\ORL\\s1\\1.pgm')

plt.imshow(img.reshape(shape), cmap='gray')
plt.show()
