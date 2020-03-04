"""
Purpose of this file is to load the MINST data file into a bunch object

This Minst dataset is the complete dataset, its is 60k images at 28x28

"""
import os.path

from matplotlib import image
from sklearn.neural_network import MLPClassifier
from sklearn.utils import Bunch
from sw_path import WORK_ROOT
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
TEST = True
LOGGING = True


print("start")



minst_training = Bunch()
minst_test = Bunch()

folder_path = (WORK_ROOT + "RES/MNIST/1kMNIST").replace("/", "\\")

if not os.path.isdir(folder_path):
    raise Exception("dir not found")

test_path = folder_path+"\\test"
training_path = folder_path+"\\training"

minst_training.data = []
minst_training.target = []
minst_training.target_names = []
minst_test.data = []
minst_test.target = []
minst_test.target_names = []
for i in range(10):
    # all image are located in dir with class name (number of the digit)
    i_dir = f"\\{i}"
    # training
    print(f"number of files in {i} dir {len(os.listdir(training_path + i_dir))}")
    # create file location
    file_location = training_path + i_dir
    # add class to the class name (called target)
    minst_training.target_names.append(i)

    # loop through the dir add each image to data
    for file in os.listdir(training_path+i_dir) :
        if file.endswith("png"):
            minst_training.data.append(image.imread(file_location+"\\"+file))
            minst_training.target.append(i)

    # testing
    print(f"number of files in {i} dir {len(os.listdir(test_path + i_dir))}")
    # create file location
    file_location = test_path + i_dir
    # add class to the class name (called target)
    minst_test.target_names.append(i)

    # loop through the dir add each image to data
    for file in os.listdir(test_path + i_dir):
        if file.endswith("png"):
            minst_test.data.append(image.imread(file_location+"\\"+file))
            minst_test.target.append(i)


if TEST:



    clf = MLPClassifier(solver='adam', max_iter=1000, alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)
    # train the classifier
    x_train = [img.flatten() for img in minst_training.data]
    x_test = [img.flatten() for img in minst_test.data]
    y_train = [minst_training.target]
    y_test = [minst_test.target]
    clf.fit(x_train, y_train)

    # find predictions
    y_pred = clf.predict(x_test)
    print(f'Accuracy is {accuracy_score(y_test, y_pred)}')
    print(f'Classification report is; \n{classification_report(y_test, y_pred)}')

print("end")
