from sklearn import datasets, neighbors, linear_model
import random
from matplotlib import pyplot as plt
print(__doc__)
'''
The iris dataset svm example is derived from;
https://scikit-learn.org/stable/auto_examples/exercises/plot_digits_classification_exercise.html#sphx-glr-auto-examples-exercises-plot-digits-classification-exercise-py
2007-2019, scikit-learn developers (BSD License)

The digit data set contains 1797 examples of handwritten digits, each are 8x8 in float ranging in values from 0 to 16
'''
# load the dataset
X_digits, y_digits = datasets.load_digits(return_X_y=True)

# Normalize the values
X_digits = X_digits / X_digits.max()

# the number of samples, which is 1797
n_samples = len(X_digits)

#split the dataset in training and test set
X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[int(.9 * n_samples):]
y_test = y_digits[int(.9 * n_samples):]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")

print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 2)



sub[0].set_title("title 1")
sub[1].set_title("title 2")

implot = plt.imshow(X_test[23].reshape(8,8))
fig.show()

