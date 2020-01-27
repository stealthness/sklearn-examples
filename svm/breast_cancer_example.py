'''
Cade based on https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

The dataset comprises 30 features (mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean
compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error,
perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error,
fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness,
worst concavity, worst concave points, worst symmetry, and worst fractal dimension) and a target (type of cancer).

This data has two types of cancer classes: malignant (harmful) and benign (not harmful)
'''
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# load the data set for breast cancer
b = datasets.load_breast_cancer()

# print the names of the features
print(f'The total number of Features : {len(b.feature_names)}')
print(f'Features names :\n {b.feature_names}\n')

# print the label types of cancer
print(f'Labels names : \n{b.target_names}\n')

# print the shape of the data
print(f'The shape of the data is {b.data.shape}')
number_of_examples = b.data.shape[0]

# Count the number of cancer of benign
print(f'The number of example that are {b.target_names[0]} are {sum(b.target)}')
print(f'The number of example that are {b.target_names[1]} are {number_of_examples - sum(b.target)}')


# split the data set into training (70%) and test set(30%)
X_train, X_test, y_train, y_test = train_test_split(b.data, b.target, test_size=0.3, random_state=109)

# Import svm from sklearn

# Create a svm Classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the model using the training data
clf.fit(X_train, y_train)

# Create some predictions using the test set
y_pred = clf.predict(X_test)

# Import sklearn.metrics to model accuracy
print(f'Accuracy is : {metrics.accuracy_score(y_test, y_pred)}')
