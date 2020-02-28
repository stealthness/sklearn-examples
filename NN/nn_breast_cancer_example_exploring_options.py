"""
This example is extension of the breast cancer example
"""
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
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

# creating a dictionarary classifications
classifiers = {}
classifiers['lbfgs_1'] = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
classifiers["ldfgs_2"] = MLPClassifier(solver='lbfgs',activation='logistic', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
classifiers["ldfgs_sgd"] = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
classifiers["ldfgs_adam"] = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
classifiers["LinSVC_2"] = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=42)
classifiers["LinSVC_4"] = OutputCodeClassifier(LinearSVC(random_state=0), code_size=4, random_state=42)
classifiers["RandomForest"] = OutputCodeClassifier(RandomForestClassifier(random_state=0), code_size=8, random_state=42)

# creating a dictionary of y_pred predictions
y_pred = {}
for k, clf in classifiers.items():
    clf.fit(x_train, y_train)
    y_pred[k] = clf.predict(x_test)

# printing out the results of predictions
for k, pred in y_pred.items():
    print(f'Result for {k}, Accuracy {accuracy_score(y_test, pred)}\n')
    print(f'Classification report is\n {classification_report(y_test, pred)}')
    print('########################################\n')

