"""
Test HoG

https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
"""
from skimage.feature import hog
from skimage import data


image = data.astronaut()

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

print(f'Original image size is {image.size}')
print(f'HoG size is {hog_image.size}')
print(f'HoG features size is {fd.size}')