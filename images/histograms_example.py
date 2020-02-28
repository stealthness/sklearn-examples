"""
This reference code for plotting histograms
"""
import numpy as np
import matplotlib.pyplot as plt


data = [0, 1, 4, 2, 3, 2, 4, 2, 1, 2, 1, 0, 1, 1]

hist = np.hstack(data)

plt.hist(hist)
plt.show()
