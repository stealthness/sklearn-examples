import numpy as np


class MeanClassifier():

    def __init__(self, **kwargs):
        self.description = 'Silly Classifier'
        self._classes = None
        self._set_of_classes = None
        self._means = None

    def fit(self, X, y):
        self._classes = y
        self._set_of_classes = set(y)
        set_list = list(self._set_of_classes)
        print(set_list)
        self._means = [None]*40
        count = [0]*40
        for i, (array_img, target) in enumerate(zip(X, y)):
            print(f'i:{i}, target: {target}')
            index = set_list.index(target)
            count[index] += 1
            if self._means[index] is None:
                self._means[index] = array_img
            else:
                self._means[index] = [a + b for (a, b) in zip(self._means[index],  array_img)]
        for i in range(len(count)):
            self._means[i] = np.array([x // count[i] for x in self._means[i]])
        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            dist = []
            for i in range(40):
            #for i, mean_img in enumerate(self._means):
                print(i)
                dist.append(np.linalg.norm(self._means[i] - x))
            #y_pred.append(list(self._set_of_classes)[dist.index(min(dist))])
            print(f' index {dist.index(min(dist))}  - {list(self._set_of_classes)[dist.index(min(dist))]}')
            y_pred.append(list(self._set_of_classes)[dist.index(min(dist))])
        return y_pred

    def get_means(self):
        return self._means
