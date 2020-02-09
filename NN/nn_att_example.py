"""
Purpos of this file is explore MLP on ATT/ORL dataset
"""
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, neighbors

# import the data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
RANDOM_STATE = 42

faces = datasets.fetch_olivetti_faces(download_if_missing=False)

# explore the data
data_size = faces.data.shape[0]
# find list of all the unique class values, then find size of this list which is 40
number_of_classes = np.unique(faces.target).size

# select a face at random
selected_face = randint(0, data_size)

plt.imshow(faces.images[selected_face], cmap='gray')
plt.show()

# We can see that the images have been normalised
v_min = faces.data.min()
v_max = faces.data.max()

print(f'The max and min values are {v_min}, {v_max} respectively')

array_of_image_aveges = np.mean(faces.data, axis=1)
print(f'If create array of the average value of each image we see that min/max vales are '
      f'{array_of_image_aveges.min():.4f}, {array_of_image_aveges.max():.4f} respectively, '
      f'therefore the images are not balance')

# split the intop training and testing sets, to make sure that we have equal amount in each classifiction we use
# option "stratify=faces.target"
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, stratify=faces.target, test_size=0.2,
                                                    random_state=RANDOM_STATE)

# show that we have 2 examples of class in the test data split
histogram_y_test = np.histogram(y_test, np.unique(faces.target))
print(f'the histogram of y_test is {histogram_y_test}')


def get_results(X_train, X_test, Y_train, **kwargs):
    # Create the classifiers
    if kwargs['clf'] == 'MLP':
        clf = MLPClassifier(solver=kwargs['solver'], max_iter=500, hidden_layer_sizes=(8, 3), random_state=RANDOM_STATE)
    elif kwargs['clf']=='NN':
        clf = KNeighborsClassifier(n_neighbors=10)
    elif kwargs['clf']=='ECOC':
        #clf = OutputCodeClassifier(LinearSVC(random_state=RANDOM_STATE, max_iter=500), code_size=20, random_state=RANDOM_STATE)
        clf = OutputCodeClassifier(estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100), code_size=20, random_state=RANDOM_STATE)
    else:
        raise Exception('No classifier selected')
    # fit the classifier
    clf.fit(X_train, Y_train)
    # return the y_pred
    return clf.predict(X_test)


y_pred = []
# y_pred.append(get_results(x_train, x_test, y_train, clf='MLP', solver='lbfgs'))
# y_pred.append(get_results(x_train, x_test, y_train, clf='MLP', solver='sgd'))
# y_pred.append(get_results(x_train, x_test, y_train, clf='MLP', solver='adam'))
# y_pred.append(get_results(x_train, x_test, y_train, clf='NN'))
y_pred.append(get_results(x_train, x_test, y_train, clf='ECOC'))

print(f'The accuracy is {accuracy_score(y_test, y_pred[0])}')
#print(f'The accuracy is {accuracy_score(y_test, y_pred[1])}')
#print(f'The accuracy is {accuracy_score(y_test, y_pred[2])}')
#print(f'The accuracy is {accuracy_score(y_test, y_pred[3])}')

print('end')
