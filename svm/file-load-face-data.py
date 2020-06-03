"""
Purpose of this file is write and test a python program for reading in face images data
"""
import os
from PIL import Image
from sw_path import WORK_ROOT as ROOT
print(__doc__)

IGNORE_FOLDERS = ["README"]


res_path = ROOT + 'RES'
att_folder = 'ORL'


def count_folders(target_path):
    """
    Returns the count of the folders in the target_path excluding files in the ignored folders
    """
    files_list = os.listdir(target_path)
    for file in IGNORE_FOLDERS:
        if file in files_list:
            files_list.remove(file)
    return len(files_list)


print(f"count is {count_folders(res_path+'/'+att_folder)}")
path = res_path+"/"+att_folder
ls = os.listdir(path)
for f in IGNORE_FOLDERS:
    if f in ls:
        ls.remove(f)

i = 0


for dirs in ls:
    count = count_folders(path+"/"+dirs)
    print(f'number of files is {count} in {path+"/"+dirs}')
    X = [[]]
    X[i] = []
    for s in os.listdir(path+"/"+dirs):
        im = Image.open(path+"/"+dirs+"/"+s)
        X[i].append(im)
