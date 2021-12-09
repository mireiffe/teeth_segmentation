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

dir_imgs = '/home/users/mireiffe/Documents/Python/ERLearning/data/er_reset/'
# dir_imgs = '/home/users/mireiffe/Documents/Python/TeethSeg/data/testimgs/'
dir_sv = '/home/users/mireiffe/Documents/Python/TeethSeg/data/trainimgs_0/'

lst_train = [[6, 8], [13, 24], [25, 39], [40, 57]]
# lst_img = [800005]

# lst_img = list(range(6, 8)) + list(range(13, 24)) + list(range(25, 39)) + list(range(40, 57))
lst_img = [39]

for li in lst_img:
    img = loadFile(join(dir_imgs, f'{li:05d}.pth'))['img']
    # img = Image.open(dir_imgs + f'{100000 + li:05d}.png')
    pimg = Image.fromarray(np.uint8(255 * img))

    # pimg = img.resize((int(img.width / 2), int(img.height / 2)))

    # print((int(img.width / 2), int(img.height / 2)))

    pimg.save(join(dir_sv, f'{li + 00000:05d}.png'), format='png')
