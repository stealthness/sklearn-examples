"""
The purpose of this script is to apply the ean classifier but after reducing the size of the image to
28 x 23. Reduceing the dimension of the
"""
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl
from sw_utils.functions import imshow_img_array

control = [True, True, True]

# grab the data (is contained in Bunch object)
b = fetch_sw_orl()

# split the data in test and train
X_train, X_test, y_train, y_true = train_test_split(b.data, b.target, test_size=0.2, stratify=b.target)

if control[0]:
    img_array = b.data[0]
    imshow_img_array(img_array, b.shape, '112x92')

    img = img_array.reshape((28,23))
    img = img.resize((28,23))