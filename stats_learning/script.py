import csv
import numpy as np

my_data = np.array([[0,0,0,0]])
with open("D:/RES/IRIS/iris.csv", "r") as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        print(line[0:4])
        if len(line) == 4:
            np.insert(my_data, line[0:4])