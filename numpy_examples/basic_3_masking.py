"""
Purpose of this file is to give examples of masking

This script is partially dirived from the LinkedIn learning course
https://www.linkedin.com/learning/numpy-data-science-essential-training/create-arrays-from-python-structures

"""
import numpy as np
# PART 2 - Boolean Mask Arrays
my_vector = np.array([-18, -4, 0, 2, 21, 37, 105])
print(f'my_vector is {my_vector}')

# create a boolean mask, based on whether an element is divisible by 3
zero_mod_3_mask = 0 == (my_vector % 3)
print(f'boolean mask is {zero_mod_3_mask}')
 #
# going create a sub array using the boolean mask
sub_array = my_vector[zero_mod_3_mask]
print(f'sub_array is {sub_array}')
# note that sub_array length is now 4
print(f'sub_array length is {sub_array.size}')

# we can also use subarray to create a new subarray
print(f'we can create sub array by using sub_array[sub_array>0] to get {sub_array[sub_array>0]}')

# another way to this
mod_test = 0 == (my_vector % 3)
pos_test = my_vector > 0
# we can use the method
combined_mask = np.logical_and(mod_test, pos_test)

print('finding sub array of elements that are %3 and >0 is {my_vector[combined_mask]}')