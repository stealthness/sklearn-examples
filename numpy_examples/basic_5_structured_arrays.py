"""
Purpose of this file is to give examples of structured arrays

This script is partially dirived from the LinkedIn learning course
https://www.linkedin.com/learning/numpy-data-science-essential-training/create-arrays-from-python-structures

"""
import numpy as np

person_data_def = [('name', 'S6'), ('height', 'f8'), ('weight', 'f8'), ('age', 'i8')]

# create a structured array
people_array = np.zeros(4, dtype=person_data_def)

print(f'The structured array is of type {type(people_array)}\n{people_array}')

# let us change some the data values
# note that any int for height or weight will processed as default
people_array[2] = ('Cat', 130, 56, 22)

people_array[0] = ('Amy', 126, 60, 25)
people_array[1] = ('Bell', 146, 60, 20)
people_array[3] = ('Amy', 140, 80, 55)

print(people_array)

# we can print the information for name, height, weight and age
ages = people_array['age']
print(f'the ages of the people are {ages}')
print(f'The names of the people are {people_array["name"]}')
print(f'The heights of the people are {people_array["height"]}')
print(f'The weights of the people are {people_array["weight"]}')

youthful = ages/2

print(f'The young ages are {youthful}')
# Note that youthful does not change the original data
print(f'The original ages are {ages}')

print(people_array[['name', 'age']])

# Record array is a thin wrapper around structured array
person_record_array = np.rec.array([('a', 100, 80, 50), ('b', 190, 189, 20)])
print(type(person_record_array[0]))