from sklearn import datasets, neighbors, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
print(__doc__)
'''
The iris dataset svm example is derived from;
https://scikit-learn.org/stable/auto_examples/exercises/plot_digits_classification_exercise.html#sphx-glr-auto-examples-exercises-plot-digits-classification-exercise-py
2007-2019, scikit-learn developers (BSD License)

The digit data set contains 1797 examples of handwritten digits, each are 8x8 in float ranging in values from 0 to 16
'''
RANDOM_STATE = 1

# load the dataset
d = datasets.load_digits()

# Normalize the values
x_digits_normalised = d.data/d.data.max()

#split the dataset in training and test set
x_train, x_test, y_train, y_test= train_test_split(x_digits_normalised, d.target, test_size=0.1, random_state=RANDOM_STATE)

# create two different classifiers
knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")

# find prediction by each classifier using x_test
y_predict_knn = knn.fit(x_train, y_train)
y_predict_logistic =  logistic.fit(x_train, y_train)

# print out the results
print(f'KNN score: {y_predict_knn.score(x_test, y_test):.2f}')
print(f'LogisticRegression score: {y_predict_logistic.score(x_test, y_test):.2f}')

# find the misidentified
misidentified = [0]*10
for x, y in zip(x_test, y_test):
    if knn.predict([x]) != y:
        misidentified[y] +=1

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(1, 2)

sub[0].set_title("Histogram of Mis identification")
sub[0].bar(d.target_names, misidentified)
sub[0].axis([-1, 10, 0, 3], sharey=True)

sub[1].set_title("Example of a Number")

sub[1].axis(visable=False)
sub[1].imshow(x_test[23].reshape(8,8))

fig.show()

