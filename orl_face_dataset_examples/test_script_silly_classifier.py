"""
The purpose of this script is to show basic outline of testing a classifier
"""
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl

# grab the data (is contained in Bunch object)
from sw_utils.silly_random_classification import SillyClassifier

b = fetch_sw_orl()

# split the data in test and train
X_train, X_test, y_train, y_true = train_test_split(b.data, b.target, test_size=0.2)

# train and predict
clf = SillyClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'the number of correct example is {accuracy_score(y_true, y_pred, normalize=False)}, with accuracy score of {accuracy_score(y_true, y_pred)}')
print(classification_report(y_true, y_pred, zero_division=0.0))
