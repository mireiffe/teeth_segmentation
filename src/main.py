import os
from os.path import join
import pickle
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage.measure import label

# custom libs
from edge_region import EdgeRegion
from balloon import Balloon
from reinitial import Reinitial


def saveFile(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def imgrad(img) -> np.ndarray:
        # ksize = 1: central, ksize = 3: sobel, ksize = -1:scharr
        gx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
        return gx / 2, gy / 2

def get_args():
    parser = argparse.ArgumentParser(description='Balloon inflated segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img", dest="imgs", nargs='+', type=int, default=[],
                             required=False, metavar="NI", 
                             help="indices of images")
    parser.add_argument("--device", dest="device", nargs='+', type=str, default=0,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("--make_er", dest="make_er", type=bool, default=False,
                             required=False, metavar="ER",
                             help="Network inference for making edge region")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    imgs = args.imgs if args.imgs else [0]

    dir_sv = 'data/netTC_210616/'

    if args.make_er:
        try:
            os.mkdir(dir_sv)
            print(f"Created save directory {dir_sv}")
        except OSError:
            pass
        for ni in imgs:
            # get edge regions from network
            edrg = EdgeRegion(args, ni)
            _img, _er = edrg.getEr()
            saveFile({'input': _img, 'output': _er}, join(dir_sv, f'T{ni:05d}.pck'))
            print(f"Edge region: {join(dir_sv, f'T{ni:05d}.pck')} is saved!!")
        os._exit(0)

    for ni in imgs:
        _dt = loadFile(join(dir_sv, f'T{ni:05d}.pck'))
        
        img = _dt['input']
        er0 = _dt['output']
        m, n, c = img.shape

        er = np.where(er0 > .5, 1., 0.)
        # can't use dilation-erosion cuz too narrow gaps
        # er = cv2.dilate(er, np.ones((5, 5)), -1, iterations=1)
        # er = cv2.erode(er, np.ones((5, 5)), -1, iterations=1)
        lbl = label(er, background=1, connectivity=1)
        del_tol = m * n / 1000
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                er[idx_i] = 1

        rein = Reinitial()
        psi = rein.getSDF(.5 - er)
        
        gx, gy = imgrad(psi)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        ec = np.where((ng < .90) * (er > .5), 1., 0.)
        ek = skeletonize(ec)

        plt.imshow(er, 'gray')
        plt.imshow(ek, alpha=.5)
        plt.show()


