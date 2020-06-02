"""
The purpose of this file is to write some code that creates basic boolean masks for 8 by 8 digit images.

the file will also print the 8 by 8 to terminal
"""
import numpy as np
from boolmask_experiment.boolean_mask import bool_not, bool_and

LOGGING = False


def mask_visual_string(mask: np.array, shape: tuple) -> str:
    """
    Returns a string suitable for terminal visualisation
    :param mask: the mask we wish to represent as a string
    :param shape: the shape of the mask
    :return: A string visualisation of the mask
    """
    sb = ""
    for row in np.reshape(mask, shape):
        element_list = []
        for b in row:
            if b:
                element_list.append("O")
            else:
                element_list.append(".")
        sb += " ".join(element_list) + "\n"
    return sb

# set constant values

ROW_DEPTH = 8
COLUMN_WIDTH = 8
S = ROW_DEPTH * COLUMN_WIDTH
digit_shape: tuple = (ROW_DEPTH, COLUMN_WIDTH)

# Test rolling a a square mask across a larger grid
sq_size = 4

sample_mask_size:tuple = (sq_size , sq_size)
initial_mask = [([True] * sq_size + [False] * (COLUMN_WIDTH - sq_size)) * sq_size + [False] * COLUMN_WIDTH * (ROW_DEPTH - sq_size)]
# print(f'initial mask is\n{mask_visual_string(initial_mask,digit_shape)}')

for j in np.arange(0, ROW_DEPTH - sq_size + 1, sq_size):
    for i in np.arange(0, COLUMN_WIDTH - sq_size + 1, sq_size):
        mask_string = mask_visual_string(np.roll(np.roll(initial_mask, i), COLUMN_WIDTH * j, axis=1), digit_shape)
        print(f'initial mask is\n{mask_string}')

# creating fixed masks

full_mask = np.array([True] * S)
empty_mask = bool_not(full_mask)

# Halves

left_half_mask = np.array(([True] * (COLUMN_WIDTH // 2) + [False] * (COLUMN_WIDTH // 2)) * (S // COLUMN_WIDTH))
right_half_mask = bool_not(left_half_mask)
middle_hoz_half_mask = np.roll(left_half_mask, 2)
top_half_mask = np.rot90(left_half_mask.reshape(digit_shape), -1).flat
bottom_half_mask = bool_not(top_half_mask)
middle_vert_Half_mask = np.roll(top_half_mask, S // 4)



# Quarters

far_left_quarter_mask = bool_and(np.roll(right_half_mask, ROW_DEPTH // 4), left_half_mask)
near_left_quarter_mask = np.roll(far_left_quarter_mask, 1 * ROW_DEPTH // 4)
near_right_quarter_mask = np.roll(far_left_quarter_mask, 2 * ROW_DEPTH // 4)
far_right_quarter_mask = np.roll(far_left_quarter_mask, 3 * ROW_DEPTH // 4)

far_top_quarter_mask = np.rot90(far_left_quarter_mask.reshape(digit_shape), -1).flat
near_top_quarter_mask = np.roll(far_top_quarter_mask, 1 * S // 4)
near_bottom_quarter_mask = np.roll(far_top_quarter_mask, 2 * S // 4)
far_bottom_quarter_mask = np.roll(far_top_quarter_mask, 3 * S // 4)

top_left_quarter_mask = bool_and(top_half_mask, left_half_mask)
top_right_quarter_mask = bool_and(top_half_mask, right_half_mask)
bottom_left_quarter_mask = bool_and(bottom_half_mask, left_half_mask)
bottom_right_quarter_mask = bool_and(bottom_half_mask, right_half_mask)

if LOGGING:
    print(f'Full mask is : \n{mask_visual_string(full_mask, digit_shape)}\n')
    print(f'Empty mask is : \n{mask_visual_string(empty_mask, digit_shape)}\n')

    print(f'left half mask is : \n{mask_visual_string(left_half_mask, digit_shape)}\n')
    print(f'right half mask is : \n{mask_visual_string(right_half_mask, digit_shape)}\n')
    print(f'top half mask is : \n{mask_visual_string(top_half_mask, digit_shape)}\n')
    print(f'bottom half mask is : \n{mask_visual_string(bottom_half_mask, digit_shape)}\n')
    print(f'middle_hoz_half_mask is : \n{mask_visual_string(middle_hoz_half_mask, digit_shape)}\n')
    print(f'middle_vert_Half_mask is : \n{mask_visual_string(middle_vert_Half_mask, digit_shape)}\n')

    print(f'far_left_quarter_mask is : \n{mask_visual_string(far_left_quarter_mask, digit_shape)}\n')
    print(f'near_left_quarter_mask is : \n{mask_visual_string(near_left_quarter_mask, digit_shape)}\n')
    print(f'far_right_quarter_mask is : \n{mask_visual_string(far_right_quarter_mask, digit_shape)}\n')
    print(f'near_right_quarter_mask is : \n{mask_visual_string(near_right_quarter_mask, digit_shape)}\n')

    print(f'far_top_quarter_mask is : \n{mask_visual_string(far_top_quarter_mask, digit_shape)}\n')
    print(f'near_top_quarter_mask is : \n{mask_visual_string(near_top_quarter_mask, digit_shape)}\n')
    print(f'far_bottom_quarter_mask is : \n{mask_visual_string(far_bottom_quarter_mask, digit_shape)}\n')
    print(f'near_bottom_quarter_mask is : \n{mask_visual_string(near_bottom_quarter_mask, digit_shape)}\n')

    print(f'top_left_quarter_mask is : \n{mask_visual_string(top_left_quarter_mask, digit_shape)}\n')
    print(f'top_right_quarter_mask is : \n{mask_visual_string(top_right_quarter_mask, digit_shape)}\n')
    print(f'bottom_left_quarter_mask is : \n{mask_visual_string(bottom_left_quarter_mask, digit_shape)}\n')
    print(f'bottom_right_quarter_mask  is : \n{mask_visual_string(bottom_right_quarter_mask, digit_shape)}\n')
