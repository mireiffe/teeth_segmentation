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
        # edrg = EdgeRegion(args, ni, scaling=False)
        edrg = EdgeRegion(args, ni, scaling=True)
        input, output = edrg.getEr()
        _dt = {'img': input, 'output': output, 'net_er': np.where(output > .5, 1., 0.)}

        self.dt.update(_dt)
        mts.saveFile(self.dt, self.path_img)

        print(f"Edge region: {self.path_img} is saved!!")
        
        self.sts.imshow(input, 'img.pdf')
        self.sts.imshow(output, 'output.pdf', cmap='gray')
        self.sts.imshow(np.where(output > .5, 1., 0.), 'net_er.pdf', cmap='gray')

    def repair_er(self):
        _dt = mts.loadFile(self.path_img)
        img = _dt['img']
        net_er = _dt['net_er']

        CP = CurveProlong(img, net_er, self.dir_save)

        _dt['er'] = CP.er
        _dt['edge_er'] = CP.edge_er
        _dt['repaired_sk'] = CP.sk
        mts.saveFile(_dt, self.path_img)
        plt.close('all')

    def seg_lvset(self):
        _dt = mts.loadFile(self.path_img)
        # seg_er = cv2.filter2D(_dt['repaired_sk'].astype(float), -1, mts.cker(3).astype(float)) > .1
        seg_er = _dt['er']
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
                plt.axis('off')
                plt.savefig(join(self.dir_save, f"test{_k:05d}.pdf"), dpi=1024, bbox_inches='tight', pad_inches=0)
            if _save or _vis:
                bln.drawContours(_k, phis, ax)
                plt.axis('off')
                # _save: plt.savefig(join(dir_resimg, f"test{_k:05d}.pdf"), dpi=1024, bbox_inches='tight', pad_inches=0)
                if _vis: plt.pause(.1)
            
            err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
            if (err < tol) or _k > max_iter:
                # new_phis[..., 0] = np.where(temp, -1., new_phis[..., 0])
                # new_phis[..., 0] = np.where(seg_er, -1., new_phis[..., 0])
                _dt['phi'] = new_phis
                mts.saveFile(_dt, self.path_img)
                break

            if _reinit:
                new_phis = np.where(new_phis < 0, -1., 1.)
                new_phis = bln.reinit.getSDF(new_phis)
            phis = new_phis
        seg_res = np.where(phis < 0, 1., 0.)
        lbl = label(seg_res, background=0, connectivity=1)
        plt.figure()
        plt.imshow(lbl)
        plt.axis('off')
        plt.savefig(f'{self.dir_save}lbl0.pdf', dpi=1024, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def post_seg(self):
        _dt = mts.loadFile(self.path_img)
        print(f'\rimage {ni}, post processing...')
        postProc = PostProc(_dt, self.dir_save)

        mts.saveFile(postProc.dict, self.path_img)


if __name__=='__main__':
    args = get_args()
    if args.ALL:
        args.make_er = True
        args.repair_er = True
        args.seg_lvset = True
        args.post_seg = True

    imgs = args.imgs if args.imgs else [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    imgs = args.imgs if args.imgs else [4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    imgs = args.imgs if args.imgs else [0, 1, 8 ,17]
    imgs = args.imgs if args.imgs else [17]

    today = time.strftime("%y%m%d", time.localtime(time.time()))
    # label_test = '1'
    label_test = None
    if label_test == None:
        dir_result = join('results', f'er_net/{today}/')
    else:
        dir_result = join('results', f'er_net/{today}_{label_test}/')
    mts.makeDir(dir_result)

    for ni in imgs:
        TSeg = TeethSeg(dir_result, ni)

        # Inference edge regions with a learned deep neural net
        if args.make_er:
            TSeg.make_er()
        
        if args.repair_er:
            TSeg.repair_er()

        if args.seg_lvset:
            TSeg.seg_lvset()

        if args.post_seg:
            TSeg.post_seg()