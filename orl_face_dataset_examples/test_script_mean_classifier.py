"""
The purpose of this script is to test MeanClassifier
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sw_utils.functions import show_class_images
from sw_utils.mean_classification import MeanClassifier
from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl

control = [True, True, True]

# grab the data (is contained in Bunch object)
b = fetch_sw_orl()

# split the data in test and train
X_train, X_test, y_train, y_true = train_test_split(b.data, b.target, test_size=0.2, stratify=b.target)

# train
clf = MeanClassifier().fit(X_train, y_train)

if control[0]:
    y_pred = clf.predict([X_test[0], X_test[1]])
    plt.imshow(X_test[0].reshape(b.shape), cmap='gray')
    plt.axis('off')
    plt.title('Test image')
    plt.show()
    show_class_images(b, y_pred[0])

if control[1]:
    # Predict
    y_pred = clf.predict(X_test)

    print(f'the number of correct example is {accuracy_score(y_true, y_pred, normalize=False)}, with accuracy score of {accuracy_score(y_true, y_pred)}')
    print(classification_report(y_true, y_pred, zero_division=0.0))

if control[2]:
    # examine mean faces
    means = clf.get_means()

    plt.imshow(means[0].reshape(b.shape), cmap='gray')
    plt.title(f'mean image for {list(b.target_list)[0]}')
    plt.show()

    show_class_images(b, list(b.target_list)[0])

print('fin')
