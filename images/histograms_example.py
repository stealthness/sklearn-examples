"""
This reference code for plotting histograms
"""
import numpy as np
import matplotlib.pyplot as plt

# Create some data
data = [0, 1, 4, 2, 3, 2, 4, 2, 1, 2, 1, 0, 1, 1]

# convert the data in an ndarray
hist = np.hstack(data)

# add the data to a histogram plt using hist() function
plt.hist(hist, align='mid')

# add a title
plt.title('An Example of Histogram Plot')

# add axis title
plt.xlabel('X axis')
plt.xticks(np.arange(0,  5, step=1))

# display Plot
plt.show()
