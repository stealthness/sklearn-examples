"""
Purpose of this file is to give examples of slicing

This script is partially dirived from the LinkedIn learning course
https://www.linkedin.com/learning/numpy-data-science-essential-training/create-arrays-from-python-structures

"""
import numpy as np

# numpy array are zero index ref
my_vector = np.array([-17, -4, 0, 2, 21, 37, 105])

# we can print the first element of the array as such
print(f'Print the first element of the array {my_vector[0]}')

# numpy array are mutable, that we can change an element
my_vector[0] = -10
print(f'numpy arrays are mutable, the first elemnt changed : {my_vector}')

# we can select any index from -[size] to [size -1], any value outside this will
# generate an error

print(my_vector[0])
print(my_vector[6])
print(my_vector[-7])

# we get some properties
print(my_vector.size)
print(my_vector.ndim)

# let create a array of 0 to 34 int array
my_array = np.arange(35)
print(f'my_array.shape is {my_array}')
# reshape it into a 2d
shape_size = (7, 5)
my_array.shape = shape_size
print(f'my_array.shape after reshaping is {my_array}')

# we can access an individual
row = 2
col = 3
print(f'The element at (2,3) is  {my_array[row, col]}')
# we can access a row of the 2d array by
print(f'The row 2 is\n{my_array[row]}\n')
# we can access a column by
print(f'The column 3 is\n {my_array[:, col]}\n')
#we can access group of columns
print(f'The column 0,2 is\n{my_array[:, [0, 2]]}\n')


# let us show the same on 3d array
print('\nA 3d example\n')
x_size = 3
y_size = 4
z_size = 5
shape_size = (x_size, y_size, z_size)
my_3d_array = np.arange(x_size*y_size*z_size)
my_3d_array.shape = shape_size
print(f'my_3d_array is \n {my_3d_array}')