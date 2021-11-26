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
from reinitial import Reinitial


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
    parser.add_argument("--trim_er", dest="trim_er",
                             required=False, action='store_true',
                             help="Trim the edge region")
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

        return 0

    def trim_er(self):
        _dt = mts.loadFile(self.path_img)
        img = _dt['img']
        net_er = _dt['net_er']

        CP = CurveProlong(img, net_er, self.dir_save)
        _dt['er'] = CP.er
        _dt['edge_er'] = CP.edge_er
        _dt['repaired_sk'] = CP.sk

        seg_er = CP.er
        lbl_reg = label(seg_er, background=1, connectivity=1)

        rein = Reinitial(dt=0.1, width=10, tol=0.01)
        phis = [rein.getSDF(np.where(lbl_reg == l, -1., 1.)) for l in range(1, np.max(lbl_reg)+1)]

        _dt.update({'seg_er': seg_er, 'phi0': phis})
        mts.saveFile(_dt, self.path_img)
        plt.close('all')

        return 0

    def post_seg(self):
        _dt = mts.loadFile(self.path_img)
        print(f'\rimage {ni}, post processing...')
        postProc = PostProc(_dt, self.dir_save, self.path_img)

        mts.saveFile(postProc.dict, self.path_img)


if __name__=='__main__':
    args = get_args()
    if args.ALL:
        args.make_er = True
        args.trim_er = True
        args.seg_lvset = True
        args.post_seg = True

    # imgs = args.imgs if args.imgs else [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    # imgs = args.imgs if args.imgs else [4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    imgs = args.imgs if args.imgs else [0, 1, 5, 8, 17, 18]
    imgs = args.imgs if args.imgs else [1]

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
        
        if args.trim_er:
            TSeg.trim_er()

        if args.post_seg:
            TSeg.post_seg()
