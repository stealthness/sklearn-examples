"""
The purpose of the file is simple demonstration of Linear Regression using sklearn functions

TO DO - INCOMPLETE
"""
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
# below code is required for 3d axes to be diplayed
from mpl_toolkits import mplot3d

print("Simple example 1")
# Create some data
x_data = np.array([[0, 0], [1, 1], [2, 2]])
y_data = np.array([0, 1, 2])

# create a linear regression model
reg = linear_model.LinearRegression()
# train it with the data
reg.fit(x_data, y_data)

print(reg.coef_)

x_min = [x_data[:,0].min(),x_data[:,1].min()]
x_max =  [x_data[:,0].max(),x_data[:,1].max()]

x_new = np.linspace(x_min, x_max, 100)
y_new = np.linspace(y_data.min(), y_data.max(), 100)

# plot scatter graph
plt.scatter(x_data[:, 0], x_data[:, 1], marker='+', color='blue')
plt.plot(x_new[:,0],x_new[:,1], color='green')
plt.xticks()
plt.show()

print("simple example in 3d scatter")
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, marker='+')
plt.plot(x_new[:,0],x_new[:,1], y_new, color='green')
plt.show()
print("end")
