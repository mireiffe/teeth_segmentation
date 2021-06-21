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
                             required=False, metavar="MER",
                             help="Network inference for making edge region")
    parser.add_argument("--repair_er", dest="repair_er", type=bool, default=False,
                             required=False, metavar="RER",
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
        
        if args.repair_er:
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
                    _dir = join('results', f'erTC_210617{ni:05d}')
                    try:
                        os.mkdir(_dir)
                        print(f"Created save directory {_dir}")
                    except OSError:
                        pass
                    if _save: plt.savefig(join(_dir, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                    if _vis: plt.pause(.1)
                
                err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
                if (err < tol) or _k > 200:
                    saveFile({'img': img, 'er': er, 'phis': new_phis}, join(_dir, f"dict{_k:05d}.pck"))
                    break

                if _reinit:
                    new_phis = np.where(new_phis < 0, -1., 1.)
                    new_phis = bln.reinit.getSDF(new_phis)
                phis = new_phis

        dt = loadFile(join('results', f'erTC_210617{ni:05d}/dict00201.pck'))
        dt = loadFile(join('results', f'ResNeSt200TC_res/test_lvset_TC00000/dict00201.pck'))
        data2 = loadFile(f'data/er_reset/{0:05d}.pth')
        img = data2['img']
        # img = dt['img']
        er = dt['er']
        phi = dt['phis'][..., 0]
        m, n = er.shape

        pre_res = np.where(phi < 0, 1., 0.)

        lbl = label(pre_res, background=0, connectivity=1)
        del_tol = m * n / 1000
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                lbl[idx_i] = 0
        non_reg_idx = np.where(lbl == 0)

        _lbl = np.zeros_like(lbl)
        for idx in zip(*non_reg_idx):
            dumy = np.zeros_like(lbl).astype(float)
            dumy[idx] = 1.
            while True:
                dumy = np.where(cv2.filter2D(dumy, -1, np.ones((3,3))) > .01, 1., 0.)
                _idx = np.where(dumy * (lbl != 0) > 0)
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
            if (sr < mu_sz - .5 * sig_sz) or sr < 50:
                sm_reg.append(i)
            elif sr > mu_sz + .8 * sig_sz:
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
        plt.subplot(2,2,1)
        plt.imshow(lbl)
        # plt.savefig(f'results/ResNeSt200TC_res/t{ni:05d}_lbl1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.figure()
        plt.subplot(2,2,2)
        plt.imshow(lbl2)
        # plt.savefig(f'results/ResNeSt200TC_res/t{ni:05d}_lbl2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.figure()
        plt.subplot(2,2,3)
        plt.imshow(res)
        # plt.savefig(f'results/ResNeSt200TC_res/t{ni:05d}_lbl3.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.figure()
        plt.subplot(2,2,4)
        plt.imshow(img)
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        for i in range(np.max(res)):
            plt.contour(np.where(res == i, -1., 1.), levels=[0], colors=clrs[i])
        # plt.savefig(f'results/ResNeSt200TC_res/t{ni:05d}_lbl3_c.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.show()

