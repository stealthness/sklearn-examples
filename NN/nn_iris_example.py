"""
This is a simple example using the iris dataset from sklearn and creating two classifiers
MLPClassifier is a Multi-layer Perceptron classifier
    see docs at https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
OutputCodeClassifier is a (Error-Correcting) Output-Code multiclass strategy
    see docs at https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html
"""
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# create a dictionary of classifiers
classifiers = {}
# creating a classification
classifiers["MLP"] = MLPClassifier(solver='lbfgs', max_iter=500, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
classifiers["LinSVC"] = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=42)

# training the classifiers
classifiers["MLP"].fit(x_train, y_train)
classifiers["LinSVC"].fit(x_train, y_train)

# creating a prediction dictionary
predictions = {}
# finding prediction using test data
predictions["MLP"] = classifiers["MLP"].predict(x_test)
predictions["LinSVC"] = classifiers["MLP"].predict(x_test)

print(f'accuracy {accuracy_score(y_test, predictions["MLP"] )}')
print(f'accuracy {accuracy_score(y_test, predictions["LinSVC"])}')

print(f'\n\nClassification report for MLPClassifier is\n {classification_report(y_test, predictions["MLP"] )}')
print(f'\n\nClassification report for MLPClassifierOutpuCodeClassifier is\n {classification_report(y_test, predictions["LinSVC"] )}')