"""
Purpose of this script is to demostrate the creation of numpy arrays

Code is partially inspired by the Linked in Learning Course
https://www.linkedin.com/learning/numpy-data-science-essential-training/create-arrays-from-python-structures

"""
import numpy as np

# check numpy version
print(f'numpy version is {np.__version__}')

# create a simply python list array
# reminder that python list are not typed bound ie my_list = [1, "1", "one", 1.0]
my_list = [1, 6, 3, -5, 10, 0, 25]

# to create a numpy array using a list
my_array_from_list = np.array(my_list)

# printing both we can se that they containing the same data
print(f'my_list is : {my_list}')
print(f'my_array_from_list is : {my_array_from_list}')

# Note that operators will behave differently from list and numpy array
print('multiply each by *3 produce different results')
print(my_list * 3)              # will treble the list by replicating the list
print(my_array_from_list *3)    # will multiple each element by 3


# if we create a tuple with different types of number, int, double and complex
my_tuple = (1, 2-2.5, 3+4j)
# the numpy array will promote all to complex so that the array elements is of the same type
print(f'numpy array from tuple is {np.array(my_tuple)}')

# Creating numpy arrays using functions

# np.arrange(n) will construct list from 0 (default) to n-1
a = np.arange(7)
print(f'np.arrange(7) produces : {a}')

# specifying a start value
b = np.arange(10,23)
print(f'np.arrange(10,23) produces : {b}')

# we find some properties of a numpy array
print(f'the length of b is {len(b)}')
print(f'the length of b is {b.size}')
print(f'the shape of b is {b.shape}')

# we can create using linspace, numpy uses [] set theory
l1 = np.linspace(5,15,9);
print(f'np.linspace(5,15,9) is {l1}')
l2 = np.linspace(5,15,9, retstep=True);
# creates array of (l1, step_value)
print(f'np.linspace(5,15,9, retstep=True) is {l2}')

# creating zeros and ones arrays
print(f'using np.zeros(5) creates {np.zeros(5)}')
# tuple size
tuple_size = (5,3)
print(f'using np.zeros((5,3)) creates\n {np.ones(tuple_size)}')
