"""
The purpose of this file is to demonstrate numpy array rotation
"""
import numpy as np


my_matrix = np.arange(18).reshape([3,6])

print(f'my_matrix is \n{my_matrix}\n')
# Note like roll() the output array doesnot affect the original array

# Note that the rotation is anticlockwise
print(f'my_matrix rot90 is \n{np.rot90(my_matrix)}\n')

# specify k = 2 that will perform 2 rot 90 (which 180)
print(f'my_matrix rot90 is \n{np.rot90(my_matrix, 2)}\n')

# by using -1 we rotated 90 clockwise
print(f'my_matrix rot90 is \n{np.rot90(my_matrix, -1)}\n')

# for 3 dim arrys we need to pecify which axis we are rotating
