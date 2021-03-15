from os.path import join

import time

import numpy as np
import pickle
import cv2


num_img = 51
dir_img = '/home/users/mireiffe/Documents/Python/TeethSeg/data/er_less'

def loadFile(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

img = loadFile(join(dir_img, f"{num_img:05d}.pth"))['img'].mean(axis=2)
layers = 2
img = np.stack([img] * layers, axis=-1)

cst = time.time()
cx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
cy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
ctt = time.time()

print(f"cv2: {ctt - cst:.5f}s")

sst = time.time()
_sx_ = img[:, 2:, ...] - img[:, :-2, ...]
x_ = img[:, 1:2, ...] - img[:, :1, ...]
_x = img[:, -1:, ...] - img[:, -2:-1, ...]
sx = np.concatenate((x_, _sx_, _x), axis=1)

_sy_ = img[2:, :, ...] - img[:-2, :, ...]
y_ = img[1:2, :, ...] - img[:1, :, ...]
_y = img[-1:, :, ...] - img[-2:-1, :, ...]
sy = np.concatenate((y_, _sy_, _y), axis=0)
stt = time.time()

print(f"slicing: {stt - sst:.5f}s")

print(np.sum(np.abs(sy - cx)))