"""
Test HoG from skimage.feature

https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
"""
from skimage.feature import hog
from skimage import data
import matplotlib.pyplot as plt

# get standard test image from skimage
image = data.astronaut()

# if we change pixels_per_cell=(32, 32) to higher values we would get fewer features
# Conversely if we reduce pixels_per_cell=(32, 32) to lower values we would get more features

# hog() returns feature vector, and hog_image if visualize=True
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

print(f'Original image size is {image.size}')
print(f'HoG size is {hog_image.size}')
print(f'HoG features size is {fd.size}')

plt.imshow(hog_image)
plt.show()

# create a figure to show images
fig = plt.figure()
# Original image subplot 1
a = fig.add_subplot(1,2,1)
img_plot = plt.imshow(image)
a.axis('off')
a.set_title('Original')

a = fig.add_subplot(1,2,2)
a.set_title('HoG')
a.axis('off')
img_plot = plt.imshow(hog_image)

fig.show()
