"""
The purpos of the file is create a bunch of boolean mask that I will then use for
"""
import numpy as np


def get_mask(size, **kwargs):
    if 'type' in kwargs:
        if kwargs['type'] == 'full':
            return np.array([True] * size)
        if kwargs['type'] == 'empty':
            return np.array([False] * size)
        if kwargs['type'] == 'half_0':
            return np.array([True] * (size // 2) + [False] * (size - size // 2))
        if kwargs['type'] == 'half_1':
            return np.array([False] * (size // 2) + [True] * (size - size // 2))
        if kwargs['type'] == 'random':
            # return get_full_mask(size)
            return np.random.rand(1, size) > 0.5
    # default, includes 'empty'
    return np.array([False] * size)


def get_mask_from_string(s: str):
    """Returns numpy boolean array created from string representation"""
    b = []
    for c in s:
        b.append(c == '1')
    return np.array(b)


def mask_to_string(mask: np.array):
    """Returns a string version of boolean np.array where True is 1, False is 0. Example [True, False, True] is '101'
    """
    sb = ""
    for b in mask:
        sb += "1" if b else "0"
    return sb