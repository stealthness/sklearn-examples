"""
Purpose of this file is to explore the bunch class from sklearn, a dictionary-like object that exposes its keys as
attributes. Documentation can be found at https://kite.com/python/docs/sklearn.utils.Bunch
"""
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from time import time

# from sklearn a typical bunch object is created
iris = datasets.load_iris()

# Description of the IRIS dataset
DESCR = iris.DESCR
print(f'The iris variable is a bunch object. It has DESCR attribute, which contains;\n{iris.DESCR}\n ')

# Features names
feature_names = iris.feature_names
number_of_features = len(feature_names)
print(f'There are {number_of_features}, these are;\n{feature_names}\n')

# target names, different classification possible
print(f'The Iris dataset main data is stored in iris.data, is has the shape {iris.data.shape}\n')
data_size = iris.data.shape[0]
print(f'That the size of the dataset is {data_size} each with {number_of_features} features\n')
d = iris.data

labels_names = iris.target_names
labels = iris.target
print(f'There are {len(labels_names)} labels, these are;\n {labels_names}\n')
print(f'There is a corresponding {len(labels)} label for each example in the dataset')

# Using the bunch object with svm


def get_result(bunch):
    time_started = time()
    x_train, x_test, y_train, y_test = train_test_split(bunch.data, bunch.target, test_size=0.5, random_state=10)
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    time_finished = time()
    return accuracy_score(y_test, y_pred), time_finished - time_started


print(f'our accuracy b is {get_result(iris)[0]}\n')

# There is better way to create a bunch object (see new_cancer)
new_iris = Bunch()
new_iris.DESCR = DESCR
new_iris.data = d
new_iris.target = labels
new_iris.target_names = labels_names
new_iris.feature_names = feature_names

result, time_taken = get_result(new_iris)
print(f'Our accuracy for new_iris is {result}\nCompleted in {1000*time_taken:.3f}\n')

cancer = datasets.load_breast_cancer()
result, time_taken = get_result(cancer)
print(f'Our accuracy for cancer dataset is {result}\nCompleted in {1000*time_taken:.3f}\n')

# Creating a new bunch object using cancer data, showing we get same result
new_cancer = Bunch(data=cancer.data, target=cancer.target)
result, time_taken = get_result(datasets.load_breast_cancer())
print(f'Our accuracy for new_cancer dataset is {result}\nCompleted in {1000*time_taken:.3f}\n')

# Apply same idea to another dataset
wine = datasets.load_wine()
result, time_taken = get_result(Bunch(data=wine.data, target=wine.target))
print(f'Our accuracy for wine dataset is {result}\nCompleted in {1000*time_taken:.3f}\n')


