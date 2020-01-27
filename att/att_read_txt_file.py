'''
This file is test reading ATT facedataset from a text file
'''
import csv
import numpy as np

filepath = 'I:/RES/ATT'
filename = 'D.txt'

f = open('I:/RES/ATT/D.txt', 'r')
print(f.readline(1))
i = 0
for line in f:
    if i > 10:
        break
    else:
        print(str(i) + " ------- " + line)

    i =i + 1

reader = csv.reader(open('I:/RES/ATT/D.txt', "r"), delimiter=",")
x = list(reader)
R = np.array(x).astype("int")