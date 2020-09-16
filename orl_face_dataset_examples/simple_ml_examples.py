"""
The purpose of this script is to demonstate some very simple example FR using simple ML models
"""
import numpy as np

from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sw_utils.mean_classification import MeanClassifier
from sw_utils.silly_random_classification import SillyClassifier
from sw_utils.functions import select_random_target, imshow_mean_img


def main():
    b_orl = fetch_sw_orl()

    #run_test_on_means(b_orl)
    run_silly(b_orl)
    #run_mean(b_orl)


def get_mean_images_of_a_target(b, target_name):
    mean_img = None
    for i, d in enumerate(b.data):
        if b.target[i] == target_name:
            if mean_img is None:
                mean_img = d
            else:
                mean_img += d
    return mean_img // 10


def get_all_mean_images_of_targets(b):
    """
    Return a list of the mean image vector of each class in b
    :param b:
    :return:
    """
    mean_images = []
    for t in b.target_list:
        mean_images.append(get_mean_images_of_a_target(b, t))
    return mean_images


def run_test_on_means(b):
    # imshow_mean_img(get_mean_images_of_a_target(b, 's1'), b.shape, 's1')

    orl_means = get_all_mean_images_of_targets(b)
    test_img, _ = select_random_target(b)

    d = []
    for img in orl_means:
        d.append(np.linalg.norm(img - test_img))

    sd = d.index(min(d))
    print(d)
    imshow_mean_img(test_img, b.shape, 'test img')
    imshow_mean_img(orl_means[sd], b.shape, ' is it?')


def run_silly(b):
    """
    Test function of SillyClassifier
    """
    print('Start Silly example')

    X_train, X_test, y_train, y_test = train_test_split(b.data, b.target, test_size=0.2, )

    clf = SillyClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    number_of_correct = []
    for (exp_label, act_label) in zip(y_train, y_pred):
        number_of_correct.append(exp_label == act_label)
        if exp_label != act_label:
            print(f'exp_label: {exp_label}, act_label : {act_label }')
        else:
            print(f'CORRECT exp_label: {exp_label}, act_label : {act_label }')
    print(number_of_correct)
    print(f'the number of correct example is {accuracy_score(y_test, y_pred, normalize=False)}, with accuracy score of {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))


def run_mean(b):
    """

    :return:
    """
    print('Start Mean example')
    print('fit')
    X_train, X_test, y_train, y_test = train_test_split(b.data, b.target, test_size=0.2, )

    clf = MeanClassifier().fit(X_train, y_train)
    print('predict')
    y_pred = clf.predict(X_test)

    number_of_correct = []
    for (exp_class, act_class) in zip(y_test, y_pred):

        number_of_correct.append(exp_class == act_class)
        print(f'exp: {exp_class} act: {act_class}')


if __name__ == "__main__":
    main()



