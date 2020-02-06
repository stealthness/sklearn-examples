"""
Purpos of this file is to use MLP and OutputCode on digits dataset
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
# load the data set
from sklearn.neural_network import MLPClassifier

digits = datasets.load_digits()

# print some details
print(f'The digits dataset contains data in the shape of {digits.data.shape}')
print(f'The digits dataset has the following classes; {digits.target_names}')

# split the data in to training and testing
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size= 0.2, random_state=42)

# create a classifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
# clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
clf = MLPClassifier(solver='adam', max_iter=1000 , alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)

# train the classifier
clf.fit(x_train, y_train)

# find predictions
y_pred = clf.predict(x_test)
print(f'Accuracy is {accuracy_score(y_test, y_pred)}')
print(f'Classification report is; \n{classification_report(y_test, y_pred)}')

hist_0 = np.histogram(digits.target, digits.target_names)
hist_1 = np.histogram(y_pred, digits.target_names)

print("""
Conclusions

solver option of 'lbfgs' and 'sgd' proved poor with accuracy of 0.3417 and 0.2056

solver option of 'adam' proved most sucessful, and impoved when max_iter was increased to 1000 from default 200. 
It had accuracy of 0.6028

""")