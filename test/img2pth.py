import os
from os.path import join

import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


dir_imgs = '/home/users/mireiffe/Documents/Python/TeethSeg/test/testimgs/'
exts = ['.jpeg', '.png', '.jpg']

ids_imgs = range(1, 11)

lst_dir = os.listdir(dir_imgs)

for file in lst_dir:
    _ni, _ext = os.path.splitext(file)
    num_img = int(_ni)

    img = plt.imread(join(dir_imgs, file))
    


    