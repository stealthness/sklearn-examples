"""
Purpose of this file is to show the use of test train splitting

"""
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
cancer = datasets.load_breast_cancer()

print(f'The breast cancer datasets contains {cancer.data.shape[0]} of examples')

# lets do a simple 70%, 30% train test split
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=1)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(f'Accuracy is : {metrics.accuracy_score(y_test, y_pred)}')

# now look at five fold cross validation
# lets do a simple 70%, 30% train test split
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=1)
