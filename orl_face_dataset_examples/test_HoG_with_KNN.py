"""
the purpose of this script is to use standarnd off the shelf HoG and KNN from scikit and apply to the ORL dataset
"""
from skimage.feature import hog
import matplotlib.pyplot as plt
from orl_face_dataset_examples.read_pgm_file import fetch_sw_orl

control = [True]

# grab the data (is contained in Bunch object)
b = fetch_sw_orl()

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

# apply HoG to all the images in b.data
hog_d = []
for img_array in b.data:
    img = img_array.reshape(b.shape)
    fd, _ = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True, multichannel=False)
    hog_d.append(fd)
b.data_hog = hog_d


