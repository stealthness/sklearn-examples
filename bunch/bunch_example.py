'''
Purpose of this file is to explore the bunch class from sklearn
'''
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from sklearn a typical bunch object is created
from sklearn.feature_selection.tests.test_rfe import test_number_of_subsets_of_features
from sklearn.utils import Bunch

b = datasets.load_iris()

# Description of the IRIS dataset
DESCR = b.DESCR
print(f'This bunch has DESCR attribute, which contains;\n{b.DESCR}\n ')

# Features names
feature_names = b.feature_names
number_of_features = len(feature_names)
print(f'There are {number_of_features}, these are;\n{feature_names}\n')

# target names, different classification possible
print(f'The Iris dataset main data is stored in b.data, is has the shape {b.data.shape}\n')
data_size = 4
print(f'That the size of the dataset is {data_size} each with {number_of_features} features\n')
d = b.data

labels_names = b.target_names
labels = b.target
print(f'There are {len(labels_names)} labels, these are;\n {labels_names}\n')
print(f'There is a corresponding {len(labels)} label for each example in the dataset')

# Using the bunch object with svm


def get_result(bunch):
    x_train, x_test, y_train, y_test = train_test_split(bunch.data, bunch.target, test_size=0.5, random_state=10)
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)


print(f'our accuracy b is {get_result(b)}\n')

# There is better way to create a bunch object (see new_cancer)
new_b = Bunch()
new_b.DESCR = DESCR
new_b.data = d
new_b.target = labels
new_b.target_names = labels_names
new_b.featur_names = feature_names

print(f'Our accuracy for new_b is {get_result(new_b)}\n')

cancer = datasets.load_breast_cancer()
print(f'value of breast cancer dataset is {get_result(cancer)}\n')

new_cancer = Bunch(data=cancer.data, target=cancer.target)
print(f'value of breast cancer dataset is {get_result(new_cancer)}\n')

wine = datasets.load_wine()
print(f'The result for wine dataset is {get_result(Bunch(data=wine.data, target=wine.target))}\n')


