import os
from os.path import join

import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def loadFile(path):
    with open(path, 'rb') as f:
        _dt = pickle.load(f)
    return _dt

dir_imgs = '/home/users/mireiffe/Documents/Python/TeethSeg/data/er_reset/'
dir_sv = '/home/users/mireiffe/Documents/Python/TeethSeg/test/testimgs/'

lst_img = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]

for li in lst_img:
    img = loadFile(join(dir_imgs, f'{li:05d}.pth'))['img']
    pimg = Image.fromarray(np.uint8(255 * img))

    pimg.save(join(dir_sv, f'{li + 100000:05d}.png'), format='png')
