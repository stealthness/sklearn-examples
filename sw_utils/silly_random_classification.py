import random


class SillyClassifier():

    def __init__(self, **kwargs):
        self.description = 'Silly Classifier'
        self._classes = None

    def fit(self, X, y):
        self._classes = y
        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(random.choice(self._classes))
        return y_pred
