'''


'''
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# creating a classification
clf_1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf_2 = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)

clf_1.fit(x_train, y_train)
clf_2.fit(x_train, y_train)

y_pred_1 = clf_1.predict(x_test)
y_pred_2 = clf_2.predict(x_test)

print(f'accuracy {accuracy_score(y_test, y_pred_1)}')
print(f'accuracy {accuracy_score(y_test, y_pred_2)}')
