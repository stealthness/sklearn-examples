"""
The purpos of the file is create a bunch of boolean mask that I will then use for
"""
import numpy as np

def get_empty_mask(size):
    return np.array([False] * size)


def get_full_mask(size):
    return np.array([True] * size)


def get_mask(size, **kwargs):
    if 'type' in kwargs:
        if kwargs['type'] == 'full':
            return get_full_mask(size)
        if kwargs['type'] == 'random':
            # return get_full_mask(size)
            return np.random.rand(1,size)>0.5
    # default, includes 'empty'
    return get_empty_mask(size)
