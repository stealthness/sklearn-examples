"""
The purpos of this file is example code of using lbp feature selection

lbp is Local Binaray Pattern

see wiki article : https://en.wikipedia.org/wiki/Local_binary_patterns
sklean : https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
"""
from matplotlib import image
from sw_path import WORK_ROOT

# load a simple image of a car

file = WORK_ROOT+'RES\TestFolder\Cars\car1.jpg'

img = image.imread(file)
