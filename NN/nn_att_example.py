"""
Purpos of this file is explore MLP on ATT/ORL dataset
"""
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# import the data
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

faces = datasets.fetch_olivetti_faces(download_if_missing=False)

# explore the data
data_size = faces.data.shape[0]
# find list of all the unique class values, then find size of this list which is 40
number_of_classes = np.unique(faces.target).size

# select a face at random
selected_face = randint(0,data_size)

plt.imshow(faces.images[selected_face], cmap='gray')
plt.show()

# We can see that the images have been normalised
v_min = faces.data.min()
v_max = faces.data.max()

print(f'The max and min values are {v_min}, {v_max} respectively')

array_of_image_aveges = np.mean(faces.data, axis=1)
print(f'If create array of the average value of each image we see that min/max vales are '
      f'{array_of_image_aveges.min():.4f}, {array_of_image_aveges.max():.4f} respectivly, '
      f'therefore the images are not balance')

# split the intop training and testing sets
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)

histogram_y_train = np.histogram(y_train,np.unique(faces.target))
histogram_y_test = np.histogram(y_test,np.unique(faces.target))
print(f'the histogram of y_train is {histogram_y_train}')
print(f'the histogram of y_test is {histogram_y_test}')

# Create the classifiers
clf = MLPClassifier(solver='lbfgs', max_iter=500, hidden_layer_sizes=(8,3), random_state=42 )
# clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=20 ,random_state=42)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(f'The accuracy is {accuracy_score(y_test, y_pred)}')


print('end')