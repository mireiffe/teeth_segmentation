import os
from os.path import join
import pickle
import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

# custom libs
from edge_region import EdgeRegion
from balloon import Balloon
from curve import CurveProlong
from post import PostProc
import myTools as mts


VISIBLE = False

def get_args():
    parser = argparse.ArgumentParser(description='Balloon inflated segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img", dest="imgs", nargs='+', type=int, default=[],
                             required=False, metavar="NI", 
                             help="indices of images")
    parser.add_argument("--device", dest="device", nargs='+', type=str, default=0,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("--make_er", dest="make_er",
                             required=False, action='store_true',
                             help="Network inference for making edge region")
    parser.add_argument("--repair_er", dest="repair_er",
                             required=False, action='store_true',
                             help="Repair the edge region")
    parser.add_argument("--seg_lvset", dest="seg_lvset",
                             required=False, action='store_true',
                             help="Segmentation by using level set method")
    parser.add_argument("--post_seg", dest="post_seg",
                             required=False, action='store_true',
                             help="Post process for segmentation")
    parser.add_argument("--ALL", dest="ALL",
                             required=False, action='store_true',
                             help="Do every process in a row")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default='cfg/inference.ini',
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()


class TeethSeg():
    def __init__(self, dir_result, num_img) -> None:
        self.dir_save = join(dir_result, f'{num_img:05d}/')
        mts.makeDir(self.dir_save)
        self.sts = mts.SaveTools(self.dir_save)
        self.path_img = join(self.dir_save, f'{num_img:05d}.pth')

        self.dt = {}
        
    def make_er(self):
        '''
        Inference edge regions with a learned deep neural net
        '''
        edrg = EdgeRegion(args, ni, scaling=False)
        input, output = edrg.getEr()
        _dt = {'img': input, 'output': output, 'net_er': np.where(output > .5, 1., 0.)}

        self.dt.update(_dt)
        mts.saveFile(self.dt, self.path_img)

        print(f"Edge region: {self.path_img} is saved!!")
        
        self.sts.imshow(input, 'img.png')
        self.sts.imshow(output, 'output.png', cmap='gray')
        self.sts.imshow(np.where(output > .5, 1., 0.), 'net_er.png', cmap='gray')

    def repair_er(self):
        _dt = mts.loadFile(path_img)
        img = _dt['img']
        net_er = _dt['net_er']

        CP = CurveProlong(img, net_er, dir_save)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        num_iter = 2
        for i in range(num_iter):
            CP.dilCurve()

            ax.cla()
            ax.imshow(img)
            ax.imshow(CP.sk, 'gray', alpha=.3)
            for idx in CP.ind_end:
                _y, _x = list(zip(*idx[::CD.gap]))
                ax.plot(_x, _y, 'r.-')
            ax.imshow(CP.new_er, alpha=.2)
            ax.set_title(f'step {i + 1}')
            if VISIBLE: plt.pause(.1)
            plt.savefig(f'{dir_save}prolong_step{i + 1}.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

            CP.reSet(k=i)

        CP.smallGap()
        _dt['er'] = CP.er
        _dt['edge_er'] = CP.edge_er
        _dt['repaired_sk'] = CP.sk
        mts.saveFile(_dt, path_img)
        plt.close('all')

    def seg_lvset(self):
        _dt = mts.loadFile(path_img)
        seg_er = cv2.dilate(_dt['repaired_sk'].astype(float), np.ones((3, 3)), -1, iterations=1)
        mgn = 2
        edge_er = np.ones_like(seg_er)
        edge_er[mgn:-mgn, mgn:-mgn] = seg_er[mgn:-mgn, mgn:-mgn]
        temp = edge_er - seg_er
        seg_er = edge_er

        bln = Balloon(seg_er, wid=5, radii='auto', dt=0.05)
        phis = bln.phis0

        _dt.update({'seg_er': seg_er, 'phi0': phis})
        # _dt.update({'seg_er': seg_er - temp, 'phi0': phis})

        fig, ax = bln.setFigure(phis)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        tol = 0.01
        _k = 0
        max_iter = 500
        _visterm = 10
        while True:
            _vis = _k % _visterm == 0 if VISIBLE else _k % _visterm == -1
            _save = _k % 3 == 3
            _k += 1
            _reinit = _k % 10 == 0

            new_phis = bln.update(phis)
            print(f"\rimage {ni}, iteration: {_k}", end='')

            if (_k == 1) or (_k > max_iter):
                bln.drawContours(_k, phis, ax)
                plt.savefig(join(dir_save, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
            if _save or _vis:
                bln.drawContours(_k, phis, ax)
                # _save: plt.savefig(join(dir_resimg, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                if _vis: plt.pause(.1)
            
            err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
            if (err < tol) or _k > max_iter:
                # new_phis[..., 0] = np.where(temp, -1., new_phis[..., 0])
                # new_phis[..., 0] = np.where(seg_er, -1., new_phis[..., 0])
                _dt['phi'] = new_phis
                mts.saveFile(_dt, path_img)
                break

            if _reinit:
                new_phis = np.where(new_phis < 0, -1., 1.)
                new_phis = bln.reinit.getSDF(new_phis)
            phis = new_phis
        seg_res = np.where(phis < 0, 1., 0.)
        lbl = label(seg_res, background=0, connectivity=1)
        plt.figure()
        plt.imshow(lbl)
        plt.savefig(f'{dir_save}lbl0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.close('all')

    def post_seg(self):
        _dt = mts.loadFile(path_img)
        print(f'\rimage {ni}, post processing...')
        postProc = PostProc(_dt, dir_save)



if __name__=='__main__':
    args = get_args()
    if args.ALL:
        args.make_er = True
        args.repair_er = True
        args.seg_lvset = True
        args.post_seg = True

    imgs = args.imgs if args.imgs else [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    # imgs = args.imgs if args.imgs else [12, 13, 14, 16, 17, 18, 20, 21]
    imgs = args.imgs if args.imgs else [0]

    today = time.strftime("%y%m%d", time.localtime(time.time()))
    # label_test = '1'
    label_test = None
    if label_test == None:
        dir_result = join('results', f'er_net/{today}/')
    else:
        dir_result = join('results', f'er_net/{today}_{label_test}/')
    mts.makeDir(dir_result)

    for ni in imgs:
        dir_save = join(dir_result, f'{ni:05d}/')
        mts.makeDir(dir_save)
        sts = mts.SaveTools(dir_save)
        path_img = join(dir_save, f'{ni:05d}.pth')
        
        # Inference edge regions with a learned deep neural net
        if args.make_er:
            edrg = EdgeRegion(args, ni, scaling=False)
            _input, _output = edrg.getEr()

            _dt = {'img': _input, 'output': _output, 'er0': np.where(_output > .5, 1., 0.)}
            mts.saveFile(_dt, path_img)
            print(f"Edge region: {path_img} is saved!!")
            
            sts.imshow(_input, 'img.png')
            sts.imshow(_output, 'output.png', cmap='gray')
            sts.imshow(np.where(_output > .5, 1., 0.), 'er0.png', cmap='gray')

        if args.repair_er:
            _dt = mts.loadFile(path_img)
            img = _dt['img']
            output = _dt['output']
            er0 = _dt['er0']

            CD = CurveProlong(er0, img, dir_save)
            num_dil = 2

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()
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
                if VISIBLE: plt.pause(.1)
                plt.savefig(f'{dir_save}prolong_step{i + 1}.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

                CD.reSet(k=i)

            CD.smallGap()
            _dt['er'] = CD.er
            _dt['edge_er'] = CD.edge_er
            _dt['repaired_sk'] = CD.sk
            mts.saveFile(_dt, path_img)
            plt.close('all')

        if args.seg_lvset:
            _dt = mts.loadFile(path_img)
            seg_er = cv2.dilate(_dt['repaired_sk'].astype(float), np.ones((3, 3)), -1, iterations=1)
            mgn = 2
            edge_er = np.ones_like(seg_er)
            edge_er[mgn:-mgn, mgn:-mgn] = seg_er[mgn:-mgn, mgn:-mgn]
            temp = edge_er - seg_er
            seg_er = edge_er

            bln = Balloon(seg_er, wid=5, radii='auto', dt=0.05)
            phis = bln.phis0

            _dt.update({'seg_er': seg_er, 'phi0': phis})
            # _dt.update({'seg_er': seg_er - temp, 'phi0': phis})

            fig, ax = bln.setFigure(phis)
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            
            tol = 0.01
            _k = 0
            max_iter = 500
            _visterm = 10
            while True:
                _vis = _k % _visterm == 0 if VISIBLE else _k % _visterm == -1
                _save = _k % 3 == 3
                _k += 1
                _reinit = _k % 10 == 0

                new_phis = bln.update(phis)
                print(f"\rimage {ni}, iteration: {_k}", end='')

                if (_k == 1) or (_k > max_iter):
                    bln.drawContours(_k, phis, ax)
                    plt.savefig(join(dir_save, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                if _save or _vis:
                    bln.drawContours(_k, phis, ax)
                    # _save: plt.savefig(join(dir_resimg, f"test{_k:05d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
                    if _vis: plt.pause(.1)
                
                err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
                if (err < tol) or _k > max_iter:
                    # new_phis[..., 0] = np.where(temp, -1., new_phis[..., 0])
                    # new_phis[..., 0] = np.where(seg_er, -1., new_phis[..., 0])
                    _dt['phi'] = new_phis
                    mts.saveFile(_dt, path_img)
                    break

                if _reinit:
                    new_phis = np.where(new_phis < 0, -1., 1.)
                    new_phis = bln.reinit.getSDF(new_phis)
                phis = new_phis
            seg_res = np.where(phis < 0, 1., 0.)
            lbl = label(seg_res, background=0, connectivity=1)
            plt.figure()
            plt.imshow(lbl)
            plt.savefig(f'{dir_save}lbl0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
            plt.close('all')

        if args.post_seg:
            _dt = mts.loadFile(path_img)
            print(f'\rimage {ni}, post processing...')
            postProc = PostProc(_dt, dir_save)

