"""
Purpose of this file is write and test a python program for reading in face images data
"""
from PIL import Image

print(__doc__)

IGNORE_FOLDES = ["README"]

import os

res_path = 'I:\RES'
att_folder = 'ATT'

def count_folders(path):
    """
    Returns the count of the folders exluding ignored folders
    """
    ls = os.listdir(path)
    for f in IGNORE_FOLDES:
        if f in ls:
            ls.remove(f)
    return len(ls)



print("count is "+ str(count_folders(res_path+"/"+att_folder)))
path = res_path+"/"+att_folder
ls = os.listdir(path)
for f in IGNORE_FOLDES:
    if f in ls:
        ls.remove(f)

i = 0
for dir in ls:
    count = count_folders(path+"/"+dir)
    print("number of files is {} in {}".format(count, path+"/"+dir))
    X = [[]]
    X[i] = []
    for s in os.listdir(path+"/"+dir):
        im = Image.open(path+"/"+dir+"/"+s)
        X[i].append(im)
