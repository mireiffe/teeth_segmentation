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
from curve import CurveDilate


def saveFile(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

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
    parser.add_argument("--repair_er", dest="repair_er", type=bool, default=False,
                             required=False, metavar="ER",
                             help="Repair the edge region")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    imgs = args.imgs if args.imgs else [1]

    dir_sv = 'data/netTC_210617/'

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

        CD = CurveDilate(np.where(er0 > .5, 1., 0.))

        # plt.figure()
        # plt.subplot(2, 3, 1)
        # plt.imshow(img)
        # plt.title('Original image')
        # # plt.subplot(2, 3, 2)
        # # plt.imshow(img)
        # # plt.imshow(er0, 'gray', alpha=.6)
        # plt.subplot(2, 3, 4)
        # plt.imshow(img)
        # plt.imshow(np.where(er0 > .5, 1., 0.), 'gray', alpha=.7)
        # plt.title('Output (thresholded)')
        # plt.subplot(2, 3, 2)
        # plt.imshow(img)
        # _er = cv2.dilate(np.where(er0 > .5, 1., 0.), np.ones((5, 5)), -1, iterations=1)
        # _er = cv2.erode(_er, np.ones((5, 5)), -1, iterations=1)
        # plt.imshow(_er, 'gray', alpha=.7)
        # plt.title('Dilation & Erosion')
        # plt.subplot(2, 3, 3)
        # plt.imshow(img)
        # plt.imshow(CD.er, 'gray', alpha=.7)
        # plt.title('Fill the holes')
        # plt.subplot(2, 3, 5)
        # plt.imshow(img)
        # plt.imshow(skeletonize(_er), 'gray', alpha=.7)
        # plt.subplot(2, 3, 6)
        # plt.imshow(img)
        # plt.imshow(CD.sk, 'gray', alpha=.7)

        num_dil = 2

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        for i in range(num_dil):
            CD.dilCurve()

            ax.cla()
            ax.imshow(img)
            ax.imshow(CD.sk, 'gray', alpha=.3)
            for idx in CD.ind_end:
                _y, _x = list(zip(*idx[::CD.gap]))
                ax.plot(_x, _y, 'r.-')
            ax.imshow(CD.new_er, alpha=.2)
            ax.set_title(f'step {i + 1}')
            plt.pause(.1)

            # CD.er = cv2.dilate(CD.er, np.ones((3, 3)), -1, iterations=1)
            CD.reSet()

            pass

        er = cv2.dilate(CD.sk.astype(float), np.ones((3, 3)), -1, iterations=1)
        bln = Balloon(er, wid=5, radii='auto', dt=0.05)
        phis = bln.phis0

        fig, ax = bln.setFigure(phis)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        tol = 0.01
        _k = 0
        while True:
            _vis = _k % 5 == 0
            _save = _k % 3 == 3

            _k += 1
            _reinit = _k % 10 == 0

            new_phis = bln.update(phis)
            print(f"\riteration: {_k}", end='')

            if _save or _vis:
                bln.drawContours(_k, phis, ax)
                _dir = join('results', f'test_lvset_TC{ni:05d}')
                try:
                    os.mkdir(_dir)
                    print(f"Created save directory {_dir}")
                except OSError:
                    pass
                if _save: plt.savefig(join(_dir, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                if _vis: plt.pause(.1)
            
            err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
            if (err < tol) or _k > 200:
                break

            if _reinit:
                new_phis = np.where(new_phis < 0, -1., 1.)
                new_phis = bln.reinit.getSDF(new_phis)
            phis = new_phis

        

