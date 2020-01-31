"""
The purpos of the file is create a bunch of boolean mask that I will then use for
"""
import numpy as np

LOGGING = False


def get_mask(size: int, **kwargs) -> np.array:
    """Returns a boolean mask which is a boolean np.array"""
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


def bool_not(arg_0):
    """Returns the logical NOT operator on a boolean numpy arrays"""
    if LOGGING:
        print(f'Logging : bool_not')
        print(f'arg_0 is {type(arg_0)} val is {arg_0}')
        print(f'return type is {type(np.logical_not(arg_0))}')
    return np.logical_not(arg_0)


def bool_and(arg_0, arg_1):
    """Returns the logical AND operator on two boolean numpy arrays"""
    if LOGGING:
        print(f'Logging : bool_and')
        print(f'arg_0 is {type(arg_0)} val is {arg_0} \n arg_1 is {type(arg_1)} val is {arg_1}')
        print(f'return type is {type(np.logical_and(arg_0, arg_1))}')
    return np.logical_and(arg_0, arg_1)


def bool_or(arg_0, arg_1):
    """Returns the logical OR operator on two boolean numpy arrays"""
    if LOGGING:
        print(f'Logging : bool_or')
        print(f'  arg_0 is {type(arg_0)} val is {arg_0} \n  arg_1 is {type(arg_1)} val is {arg_1}')
        print(f'return type is {type(np.logical_or(arg_0, arg_1))}')
    return np.logical_or(arg_0, arg_1)


def bool_xor(arg_0, arg_1):
    """Returns the logical XOR operator on two boolean numpy arrays"""
    if LOGGING:
        print(f'Logging : bool_xor')
        print(f'  arg_0 is {type(arg_0)} val is {arg_0} \n  arg_1 is {type(arg_1)} val is {arg_1}')
        print(f'return type is {type(np.logical_xor(arg_0, arg_1))}')
    return np.logical_xor(arg_0, arg_1)