from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import watershed_ift
import torch.nn.functional as F
import torch
import pickle
import cv2

from reinitial import Reinitial

from scipy.io import savemat, loadmat

# num_img = [0, 1, 5, 8, 17, 18]
num_img = [13]

rein = Reinitial(width=2)

for ni in num_img:
    # dt = loadmat(f'/home/users/mireiffe/Documents/Python/TeethSeg/otherpapers/Lee2010_MCWA/{300000 + ni}.mat') 
    dt = loadmat(f'/home/users/mireiffe/Documents/Python/TeethSeg/otherpapers/Na2014_MWA/{400000 + ni}.mat') 

    img = dt['img']
    # res = dt['L']

    _res = dt['res']
    bb = dt['coord'][0]
    res = -np.ones_like(img.mean(2))
    res[bb[2]-1:bb[3], bb[0]-1:bb[1]] = _res

    plt.figure()
    plt.imshow(img)

    for l in np.unique(res):
        if l in [0, -1]: continue

        _reg = -1 * (res == l) + 1 * (res != l) * (res != 0)
        _phi = rein.getSDF(_reg)
    
        plt.contour(_phi, levels=[0], colors='lime', linewidths=1.2)
    #plt.show()
    plt.axis('off')
    # plt.savefig(f'/home/users/mireiffe/Documents/Python/TeethSeg/otherpapers/Lee2010_MCWA/{300000+ni}.pdf', dpi=256, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'/home/users/mireiffe/Documents/Python/TeethSeg/otherpapers/Na2014_MWA/{400000+ni}.pdf', dpi=64, bbox_inches='tight', pad_inches=0)
