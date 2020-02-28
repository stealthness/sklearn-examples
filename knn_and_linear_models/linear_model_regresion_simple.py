"""
The purpose of the file is simple demonstration of Linear Regression using sklearn functions
"""
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
print("start")
reg = linear_model.LinearRegression()
x_data = np.array([[0, 0],[1, 1],[2, 2]])
y_data = [0,1,2]
reg.fit(x_data, y_data)
print(reg.coef_)

x_new = np.linspace(x.min, x_max,0 )

# plot scatter graph
plt.scatter(x_data[:, 0], x_data[:, 1], marker='+', color='blue')
plt.plot()
plt.xticks()
plt.show()
print("end")
