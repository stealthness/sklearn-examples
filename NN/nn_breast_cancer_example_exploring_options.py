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

# creating a classification
clf_1_1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf_1_2 = MLPClassifier(solver='lbfgs',activation='logistic', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf_1_3 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf_1_4 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf_2_1 = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=42)
clf_2_2 = OutputCodeClassifier(LinearSVC(random_state=0), code_size=4, random_state=42)
clf_2_3 = OutputCodeClassifier(RandomForestClassifier(random_state=0), code_size=8, random_state=42)

classifiers = [clf_1_1, clf_1_2, clf_1_3 , clf_1_4 , clf_2_1 , clf_2_2 , clf_2_3]
y_pred = []
for clf in classifiers:
    clf.fit(x_train, y_train)
    y_pred.append(clf.predict(x_test))

for pred in y_pred:
    print(f'Accuracy {accuracy_score(y_test, pred)}')
    print(f'\n\nClassification report is\n {classification_report(y_test, pred)}')
    print('########################################')

