"""
the purpose of this script is to use standarnd off the shelf HoG and KNN from scikit and apply to the ORL dataset
"""
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
import time
from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl

control = [False, True]

# grab the data (is contained in Bunch object)
b = fetch_sw_orl()
tic = time.time()
if control[0]:
    # hog() returns feature vector, and hog_image if visualize=True
    # apply this to the first image of b.data
    fd, hog_image = hog(b.data[0].reshape(b.shape), orientations=8, pixels_per_cell=(32, 32),
                        cells_per_block=(1, 1), visualize=True, multichannel=False)

    print(f'Original image size is {b.shape}')
    print(f'HoG size is {hog_image.size}')
    print(f'HoG features size is {fd.size}')
    fig = plt.figure()
    fig.add_subplot(1,2, 1)
    plt.imshow(b.data[0].reshape(b.shape), cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.show()

# split the data in
X_train, X_test, y_train, y_true = train_test_split(b.data, b.target, test_size=0.2, stratify=b.target)



# apply HoG to all the images in b.data
hog_train = []
for img_array in X_train:
    img = img_array.reshape(b.shape)
    fd, _ = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True, multichannel=False)
    hog_train.append(fd)


clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=42)
clf.fit(hog_train, y_train)
tok = time.time()
if control[1]:
    # create the hog fro the X_test
    hog_test = []
    for img_arry in X_test:
        fd, _ = hog(img_arry.reshape(b.shape), orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True, multichannel=False)
        hog_test.append(fd)
    y_pred = clf.predict(hog_test)

    print(f'the number of correct example is {accuracy_score(y_true, y_pred, normalize=False)}, with accuracy score of {accuracy_score(y_true, y_pred)}')
    print(classification_report(y_true, y_pred, zero_division=0.0))
    print(f'time to train : {tok - tic:.5}')





