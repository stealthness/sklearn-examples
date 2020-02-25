"""
Purpose of this file is to give examples of masking

This script is partially dirived from the LinkedIn learning course
https://www.linkedin.com/learning/numpy-data-science-essential-training/create-arrays-from-python-structures

"""
import numpy as np

a1 = np.arange(70)
a2 = np.arange(3*4*5)

a1.shape = (2, 7, 5)
print(f"a1 is \n {a1}\n")
print(f"a2 is \n {a2}\n")

# shape attribute will return size of each dimension
# ndim attribute will return the number of dimension
# size attribute will return the number opf elements
# dtype is the type of of the elements in the array (which must all be same)
print(f'a1 length of each dimension is = {a1.shape}')
print(f'a1 has the number of dimensions = {a1.ndim}')
print(f'a1 the total number of elements is =  {a1.size}')
print(f'a1 elements is of type =  {a1.dtype} \n')


print(f'applying 5 * a1 - 2 is \n {5 * a1 - 2}')

# to do
left_mat = np.arange(6).reshape((2, 3))
right_mat = np.arange(15).reshape((3, 5))
print(f'left_mat is\n{left_mat}\n')
print(f'right_mat is\n{right_mat}\n')

# take the inner product
#  np.inner(left_mat, right_mat) will not work as there is a direction mismatch inner used for 1d array
# for 2d+ array use np.dot
print(f'the dot problem \n{np.dot(left_mat, right_mat)}\n')

# we sum the array
print(f'The sum of all the elements in a1 is {a1.sum()}')