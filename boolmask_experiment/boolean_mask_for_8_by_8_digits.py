"""
The purpose of this file is to write some code that creates basic boolean masks for 8 by 8 digit images.

the file will also print the 8 by 8 to terminal
"""
import numpy as np

from boolmask_experiment.boolean_mask import bool_not, bool_and


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
    N = 8
    M = 12
    S = N*M
    digit_shape = [8, 12]

    full_mask = np.array([True] * S)
    print(f'Full mask is : \n{mask_visual_string(full_mask, digit_shape)}\n')

    empty_mask = bool_not(full_mask)
    print(f'Empty mask is : \n{mask_visual_string(empty_mask, digit_shape)}\n')

    left_half_mask = np.array(([True]*(M//2) +[False]*(M//2))*(S//M))
    right_half_mask = bool_not(left_half_mask)
    middle_hoz_half_mask = np.roll(left_half_mask, 2)

    top_half_mask = np.rot90(left_half_mask.reshape(digit_shape),-1).flat
    bottom_half_mask = bool_not(top_half_mask)
    middle_vert_Half_mask = np.roll(top_half_mask, S//4)

    print(f'left half mask is : \n{mask_visual_string(left_half_mask, digit_shape)}\n')
    print(f'right half mask is : \n{mask_visual_string(right_half_mask, digit_shape)}\n')
    print(f'top half mask is : \n{mask_visual_string(top_half_mask, digit_shape)}\n')
    print(f'bottom half mask is : \n{mask_visual_string(bottom_half_mask, digit_shape)}\n')
    print(f'middle_hoz_half_mask is : \n{mask_visual_string(middle_hoz_half_mask, digit_shape)}\n')
    print(f'middle_vert_Half_mask is : \n{mask_visual_string(middle_vert_Half_mask, digit_shape)}\n')

    far_left_quarter_mask = bool_and(np.roll(right_half_mask, N//4), left_half_mask)
    near_left_quarter_mask = np.roll(far_left_quarter_mask, 1*N//4)
    near_right_quarter_mask = np.roll(far_left_quarter_mask, 2*N//4)
    far_right_quarter_mask = np.roll(far_left_quarter_mask, 3*N//4)
    
    print(f'far_left_quarter_mask is : \n{mask_visual_string(far_left_quarter_mask, digit_shape)}\n')
    print(f'near_left_quarter_mask is : \n{mask_visual_string(near_left_quarter_mask, digit_shape)}\n')
    print(f'far_right_quarter_mask is : \n{mask_visual_string(far_right_quarter_mask, digit_shape)}\n')
    print(f'near_right_quarter_mask is : \n{mask_visual_string(near_right_quarter_mask, digit_shape)}\n')

    far_top_quarter_mask = np.rot90(far_left_quarter_mask.reshape(digit_shape),-1).flat
    near_top_quarter_mask = np.roll(far_top_quarter_mask, 1*S//4)
    near_bottom_quarter_mask = np.roll(far_top_quarter_mask, 2*S//4)
    far_bottom_quarter_mask = np.roll(far_top_quarter_mask, 3*S//4)

    print(f'far_top_quarter_mask is : \n{mask_visual_string(far_top_quarter_mask, digit_shape)}\n')
    print(f'near_top_quarter_mask is : \n{mask_visual_string(near_top_quarter_mask, digit_shape)}\n')
    print(f'far_bottom_quarter_mask is : \n{mask_visual_string(far_bottom_quarter_mask, digit_shape)}\n')
    print(f'near_bottom_quarter_mask is : \n{mask_visual_string(near_bottom_quarter_mask, digit_shape)}\n')

    top_left_quarter_mask = bool_and(top_half_mask,left_half_mask)
    top_right_quarter_mask = bool_and(top_half_mask,right_half_mask)
    bottom_left_quarter_mask = bool_and(bottom_half_mask,left_half_mask)
    bottom_right_quarter_mask = bool_and(bottom_half_mask,right_half_mask)

    print(f'top_left_quarter_mask is : \n{mask_visual_string(top_left_quarter_mask, digit_shape)}\n')
    print(f'top_right_quarter_mask is : \n{mask_visual_string(top_right_quarter_mask, digit_shape)}\n')
    print(f'bottom_left_quarter_mask is : \n{mask_visual_string(bottom_left_quarter_mask, digit_shape)}\n')
    print(f'bottom_right_quarter_mask  is : \n{mask_visual_string(bottom_right_quarter_mask , digit_shape)}\n')

if __name__ == "__main__":
    main()
