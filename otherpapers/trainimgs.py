import pickle
import myTools as mts
import numpy as np


dir_dt = '/home/users/mireiffe/Documents/Python/TeethSeg/data/er_reset/'
dir_sv = '/home/users/mireiffe/Documents/Python/TeethSeg/data/trainimgs_MRCNNlabel/'
num_img = [6, 7] + list(range(13, 57))

for ni in num_img:
    dt = mts.loadFile(dir_dt + f'{ni:05d}.pth')
    img = dt['img']


    mts.imwrite(img, dir_sv + f'{ni:05d}.png')
