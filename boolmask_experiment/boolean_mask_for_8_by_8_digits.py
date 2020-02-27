"""
The purpose of this file is to write some code that creates basic boolean masks for 8 by 8 digit images.

the file will also print the 8 by 8 to terminal
"""
import numpy as np

from boolmask_experiment.boolean_mask import bool_not


def print_mask(mask, shape):
    print(mask_visual_string(mask, shape))


def mask_visual_string(mask, shape):
    sb = ""
    for row in np.reshape(mask, shape):
        list = []
        for b in row:
            if b:
                list.append("O")
            else:
                list.append(".")
        sb += " ".join(list) + "\n"
    return sb


def main():
    digit_shape = [8, 8]
    full_mask = np.array([True] * 64)
    print(f'Full mask is : \n{mask_visual_string(full_mask, digit_shape)}\n')

    empty_mask = np.array([False] * 64)
    print(f'Empty mask is : \n{mask_visual_string(empty_mask, digit_shape)}\n')

    left_half_mask = np.array(([True]*4 +[False]*4)*8)
    print(f'left half mask is : \n{mask_visual_string(left_half_mask, digit_shape)}\n')

    right_half_mask = bool_not(left_half_mask)
    print(f'right half mask is : \n{mask_visual_string(right_half_mask, digit_shape)}\n')

    top_half_mask = np.array([True]*32 +[False]*32)
    print(f'top half mask is : \n{mask_visual_string(top_half_mask, digit_shape)}\n')

    bottom_half_mask = bool_not(top_half_mask)
    print(f'bottom half mask is : \n{mask_visual_string(bottom_half_mask, digit_shape)}\n')




if __name__ == "__main__":
    main()
