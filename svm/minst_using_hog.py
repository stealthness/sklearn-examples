'''
Using code learnt from https://www.datacamp.com/community/tutorials



'''
from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# import minst dataset
b = datasets.load_digits()

# print out the number and labels name
print(f'There are {len(b.target_names)} labels names')
print(f'These names are :\n {b.target_names}\n')

number_of_examples = b.data.shape[0]
print(f'The number of examples are {number_of_examples}')

# Show a random single image
random_selection = random.randint(0, number_of_examples)
img = b.data[random_selection].reshape([8, 8]) / 16
col_map = plt.get_cmap('gray')
plt.imshow(img, cmap=col_map)
plt.show()

# select just a few pixels create a random boolean mask
boolean_mask = np.random.choice(a=[False, True], size=64)


def get_mod_x(data,mask):
    mod_x = []
    for d in data:
        mod_x.append(d[mask])
    return mod_x


def do_training_and_predicting(data, labels):
    # split the data set into training (80%) and test set(20%)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=109)
    clf = svm.SVC(gamma='auto')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # Import sklearn.metrics to model accuracy
    return metrics.accuracy_score(y_test, y_pred)


do_training_and_predicting(b.data, b.target)
do_training_and_predicting(get_mod_x(b.data, boolean_mask), b.target)


# try do GA population masks Initialise
pop_size = 10
elitism = 5
count = 0
solution_found = False
population = []
pop_X = []
for i in range(pop_size):
    population.append(np.random.choice(a=[False, True], size=64))

while not solution_found and count < 10:
    result = []
    for i in range(10):
        result.append(do_training_and_predicting(get_mod_x(b.data, population[i]), b.target))

    # we keep top
    elitism_idx = np.argsort(result[-elitism:])
    new_population = []
    for i in elitism_idx:
        new_population.append(population[i])
    for i in range(pop_size - elitism):
        new_population.append(np.random.choice(a=[False, True], size=64))
    population = new_population
    print(f' best result = {result[np.argsort(result)[-1]]}')
    print(f'\n\n generation {count}\n\n')
    count = count + 1
