# 检测npy文件

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import shutil
import glob

def check_npy_file(npy_path):
    data = np.load(npy_path)
    new_data = np.empty((30, 2), dtype=object)
    for row, line in enumerate(data):
        new_data[row][0] = line[0]
        new_data[row][1] = line[1]
    print(new_data)
    np.save('/data2/zwz/PycharmProject/FOCAL/flist/DVI_tra_30.npy', new_data)

if __name__ == '__main__':
    path = '/data2/zwz/PycharmProject/FOCAL/flist/DVI_tra_30.npy'
    check_npy_file(path)