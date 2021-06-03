import os
from os.path import join
import argparse

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import matplotlib.animation as animation

# custom libs
from edge_region import EdgeRegion
from balloon import Balloon


tol = 0.01
dir_save = '/home/users/mireiffe/Documents/Python/TeethSeg/results'

def get_args():
    parser = argparse.ArgumentParser(description='Balloon inflated segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img", dest="num_img", type=int, default=False,
                             required=True, metavar="NI", 
                             help="number of image")
    parser.add_argument("--device", dest="device", nargs='+', type=str, default=False,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=True, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

def savepck(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

if __name__=='__main__':
    args = get_args()

    # num_imgs = range(1, 2)
    # for ni in num_imgs:
    #     # get edge regions from network
    #     edrg = EdgeRegion(args.path_cfg, ni)
    #     er = edrg.getEr()
    #     savepck(er, f"data/net_ers_TC/T{ni:05d}.pck")
    # os._exit(0)

    ## get edge regions from network
    # edrg = EdgeRegion(args.path_cfg, args.num_img)
    # er = edrg.getEr()

    # plt.figure()
    # plt.imshow(er, 'gray')
    # plt.show()

    if args.num_img < 0:
        _sz = 128, 128
        _c = 64, 64
        _r = 20
        ang = 20 * np.pi / 180        # degree

        [X, Y] = np.indices((_sz[0], _sz[1]))
        cdt1 = (X - _c[0])**2 + (Y - _c[1])**2 < _r**2
        cdt2 = (X - _c[0])**2 + (Y - _c[1])**2 >= (_r - 2)**2
        er = np.where(cdt1 * cdt2, 1., 0.)
        er = np.where(((Y - _c[1]) - np.tan(ang) * (X - _c[0]) < 0) * ((Y - _c[1]) + np.tan(ang) * (X - _c[0])) > 0, 0., er)

        er_ = np.zeros_like(er)
        # for x in zip(range(61, 67), range(61, 67)):
        #     er_[x[0], x[1]] = 1
        # for x in zip(range(62, 68), range(61, 67)):
        #     er_[x[0], x[1]] = 1
        # er_ = cv2.dilate(er_, np.ones((3, 3)), iterations=1)
        
        er = er + er_

    for ni in [1]:
        # er = cv2.dilate(loadFile('results/er_test.pck'), np.ones((3,3)), iterations=1)
        er = loadFile(f'results/er_dil_iter5_TC/er_test{ni:05d}.pck')
        er = skeletonize(er)
        er = cv2.dilate(np.where(er > .5, 1., 0.), np.ones((3, 3)), iterations=1)[2:-2, 2:-2]

        bln = Balloon(args.num_img, er, wid=5, radii='auto', dt=0.03)
        phis = bln.phis0

        # FOR TEST!!!!!!!!!!!!!!
        # inits = bln.getInitials()
        # inits = np.expand_dims(np.where((X - 84)**2 + (Y-64)**2 < 10**2, 1., inits[..., 0]), axis=-1)
        # phis = bln.reinit.getSDF(inits)
        # END!!!!!!!!!!!!!!!!!

        fig, ax = bln.setFigure(phis)
        mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        # mng.resize(*mng.window.maxsize())
        mng.full_screen_toggle()
        
        _k = 0
        while True:
            _vis = _k % 10 == 0
            _save = _k % 3 == 4

            _k += 1
            _reinit = _k % 10 == 0

            new_phis = bln.update(phis)
            print(f"\riteration: {_k}", end='')

            if _save or _vis:
                bln.drawContours(_k, phis, ax)
                _dir = join(dir_save, f'test_lvset_TC{ni:05d}')
                try:
                    os.mkdir(_dir)
                    print(f"Created save directory {_dir}")
                except OSError:
                    pass
                if _save: plt.savefig(join(_dir, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                if _vis: plt.pause(.5)
            
            err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
            if (err < tol) or _k > 300:
                break

            if _reinit:
                new_phis = np.where(new_phis < 0, -1., 1.)
                new_phis = bln.reinit.getSDF(new_phis)
            phis = new_phis
            
        savepck({'er': er, 'phis': new_phis}, join(_dir, f"dict{_k:05d}.pck"))
