"""
Taking the example from nn_iris_example.py and applying to the breast cancer dataset
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report

print(__doc__)

# load the breast cancer dataset
breast = datasets.load_breast_cancer()

#check details
print(f'The size of the data is {breast.data.shape}')
print(f'There are {breast.target_names} classifiers')

# split the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(breast.data, breast.target, test_size=0.2)

# creating a classification
clf_1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf_2 = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=42)

clf_1.fit(x_train, y_train)
clf_2.fit(x_train, y_train)

y_pred_1 = clf_1.predict(x_test)
y_pred_2 = clf_2.predict(x_test)

print(f'accuracy {accuracy_score(y_test, y_pred_1)}')
print(f'accuracy {accuracy_score(y_test, y_pred_2)}')

print(f'\n\nClassification report for MLPClassifier is\n {classification_report(y_test, y_pred_2)}')
print(f'\n\nClassification report for MLPClassifierOutpuCodeClassifier is\n {classification_report(y_test, y_pred_2)}')