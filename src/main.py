import os
from os.path import join
import pickle
import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage.measure import label

# custom libs
from edge_region import EdgeRegion
from balloon import Balloon
from reinitial import Reinitial
from curve import CurveProlong
from post import PostProc


def saveFile(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def makeDir(path):
    try:
        os.mkdir(path)
        print(f"Created a directory {path}")
    except OSError:
        pass

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
                             required=False, metavar="MER", action='store_true',
                             help="Network inference for making edge region")
    parser.add_argument("--repair_er", dest="repair_er", type=bool, default=False,
                             required=False, metavar="RER", action='store_true',
                             help="Repair the edge region")
    parser.add_argument("--seg_lvset", dest="seg_lvset", type=bool, default=False,
                             required=False, metavar="SEG", action='store_true',
                             help="Segmentation by using level set method")
    parser.add_argument("--post_seg", dest="post_seg", type=bool, default=False,
                             required=False, metavar="PST", action='store_true',
                             help="Post process for segmentation")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    imgs = args.imgs if args.imgs else [2, 3, 4, 5, 8, 9, 10, 11, 12]

    current_time = time.strftime("%y%m%d", time.localtime(time.time()))
    
    dir_er = 'data/netTC_210617/'
    dir_result = join('results', f'er_net/{current_time}/')

    makeDir(dir_result)
    makeDir(dir_er)

    for ni in imgs:
        dir_resimg = join(dir_result, f'{ni:05d}/')
        makeDir(dir_resimg)
        
        # Inference edge regions with a learned deep neural net
        if args.make_er:
            path_er = join(dir_er, f'T{ni:05d}.pck')
            edrg = EdgeRegion(args, ni)
            _img, _er = edrg.getEr()

            saveFile({'input': _img, 'output': _er}, path_er)
            print(f"Edge region: {path_er} is saved!!")
            continue

        _dt = loadFile(join(dir_er, f'T{ni:05d}.pck'))
        if args.repair_er:
            img = _dt['input']
            er0 = _dt['output']

            plt.figure()
            plt.imshow(img)
            plt.savefig(f'{dir_resimg}img.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
            plt.figure()
            plt.imshow(er0, 'gray')
            plt.savefig(f'{dir_resimg}er0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

            CD = CurveProlong(np.where(er0 > .5, 1., 0.), img)
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

                CD.reSet()
            continue

        if args.seg_lvset:
            er = cv2.dilate(CD.sk.astype(float), np.ones((3, 3)), -1, iterations=1)
            bln = Balloon(er, wid=5, radii='auto', dt=0.05)
            phis = bln.phis0

            fig, ax = bln.setFigure(phis)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            
            tol = 0.01
            _k = 0
            max_iter = 300
            while True:
                _vis = _k % 5 == 0
                _save = _k % 3 == 3

                _k += 1
                _reinit = _k % 10 == 0

                new_phis = bln.update(phis)
                print(f"\riteration: {_k}", end='')

                if _save or _vis:
                    bln.drawContours(_k, phis, ax)
                    if _save: plt.savefig(join(dir_resimg, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                    if _vis: plt.pause(.1)
                
                err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
                if (err < tol) or _k > max_iter:
                    saveFile({'img': img, 'er': er, 'phis': new_phis}, join(dir_resimg, f"dict.pck"))
                    break

                if _reinit:
                    new_phis = np.where(new_phis < 0, -1., 1.)
                    new_phis = bln.reinit.getSDF(new_phis)
                phis = new_phis
            continue

        if args.post_seg:
            postProc = PostProc(dir_resimg)

