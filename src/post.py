import os 

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label


def loadFile(path):
    with open(path, 'rb') as f:
        _dt = pickle.load(f)
    return _dt


num_img = 55

path_dt = f'results/test_lvset{num_img:05d}/dict00151.pck'
data = loadFile(path_dt)

data2 = loadFile(f'data/er_less/{num_img:05d}.pth')
img = data2['img']

er = data['er']
phi = data['phis'][..., 0]

pre_res = np.where(phi < 0, 1., 0.)

lbl = label(pre_res, background=0, connectivity=1)

non_reg_idx = np.where(phi >= 0)

_lbl = np.zeros_like(lbl)
for idx in zip(*non_reg_idx):
    dumy = np.zeros_like(lbl).astype(float)
    dumy[idx] = 1.
    while True:
        dumy = np.where(cv2.filter2D(dumy, -1, np.ones((3,3))) > .01, 1., 0.)
        _idx = np.where(dumy * pre_res > 0)
        if len(_idx[0]) > 0:
            _lbl[idx[0], idx[1]] = lbl[_idx[0][0], _idx[1][0]]
            break

lbl2 = lbl + _lbl

num_reg = np.max(lbl2) + 1
sz_reg = [np.sum(lbl2 == i + 1) for i in range(num_reg)]
mu_sz = sum(sz_reg) / num_reg
mu_sq = sum([sr ** 2 for sr in sz_reg]) / num_reg
sig_sz = np.sqrt(mu_sq - mu_sz ** 2)

sm_reg = []
lg_reg = []
for i, sr in enumerate(sz_reg):
    if sr < mu_sz - .5 * sig_sz:
        sm_reg.append(i)
    elif sr > mu_sz + 3 * sig_sz:
        lg_reg.append(i)

lbl3 = np.copy(lbl2)
idx2 = [np.where(lbl2 == i + 1) for i in range(num_reg)]

for lr in lg_reg[1:]:
    lbl3[idx2[lr]] = lg_reg[0] + 1

# _lbl = np.zeros_like(lbl3)
# non_reg_idx2 = []
# for sr in sm_reg[1:]:
#     non_reg_idx2.append(np.where(lbl3 == sr))

# tt = np.ones_like(lbl3)
# for nri in non_reg_idx2:
#     for idx in zip(*nri):
#         tt[idx] = 0

# for nri in non_reg_idx2:
#     for idx in zip(*nri):
#         dumy = np.zeros_like(lbl).astype(float)
#         dumy[idx] = 1.
#         while True:
#             dumy = np.where(cv2.filter2D(dumy, -1, np.ones((3,3))) > .01, 1., 0.)
#             _idx = np.where(dumy * tt > 0)
#             if len(_idx[0]) > 0:
#                 _lbl[idx[0], idx[1]] = lbl3[_idx[0][0], _idx[1][0]]
#                 break

lbl4 = lbl3

res = lbl4
plt.figure()
plt.imshow(lbl)
plt.savefig(f'results/t{num_img:05d}_lbl1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
plt.figure()
plt.imshow(lbl2)
plt.savefig(f'results/t{num_img:05d}_lbl2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
plt.figure()
plt.imshow(res)
plt.savefig(f'results/t{num_img:05d}_lbl3.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
plt.figure()
plt.imshow(img)
clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
for i in range(np.max(res)):
    plt.contour(np.where(res == i, -1., 1.), levels=[0], colors=clrs[i])
plt.savefig(f'results/t{num_img:05d}_lbl3_c.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
plt.show()

pass
