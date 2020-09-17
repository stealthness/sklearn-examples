"""
The purpose of this script is to show basic outline of testing a classifier
"""
import time

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl

# grab the data (is contained in Bunch object)
from sw_utils.mean_classification import MeanClassifier
from sw_utils.silly_random_classification import SillyClassifier


def run_test(c, b):
    # split the data in test and train
    X_train, X_test, y_train, y_true = train_test_split(b.data, b.target, test_size=0.2)

    # train and predict
    tic = time.time()
    clf = c.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    toc = time.time()

    result = accuracy_score(y_true, y_pred, normalize=True)
    report = classification_report(y_true, y_pred, zero_division=0.0)

    return result, report, toc - tic


def print_info(args):
    for r in args:
        print(r)

def print_acur(args):
    print(f'{args[0]}%, in {args[2]}')

# load data
b_orl = fetch_sw_orl()


# silly
silly_classifier = SillyClassifier()
print_acur(run_test(silly_classifier, b_orl))

# mean
mean_classifier = MeanClassifier()
print_acur(run_test(mean_classifier, b_orl))

