from os.path import join
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

# custom libs
from edge_region import EdgeRegion
from refine import RefinePreER
from snake import Snake
from idreg import IdRegion
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
    parser.add_argument("--pre_er", dest="pre_er",
                             required=False, action='store_true',
                             help="Network inference for making edge region")
    parser.add_argument("--refine", dest="refine",
                             required=False, action='store_true',
                             help="Refinment of regions")
    parser.add_argument("--snake", dest="snake",
                             required=False, action='store_true',
                             help="snake")
    parser.add_argument("--id_reg", dest="id_reg",
                             required=False, action='store_true',
                             help="Idetification of regions")
    parser.add_argument("--ALL", dest="ALL",
                             required=False, action='store_true',
                             help="Do every process in a row")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default='cfg/inference.ini',
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

class TeethSeg():
    def __init__(self, dir_result, num_img) -> None:
        print(f'image {ni}, initiating...')

        self.dir_save = join(dir_result, f'{num_img:05d}/')
        mts.makeDir(self.dir_save)
        self.sts = mts.SaveTools(self.dir_save)
        self.path_dict = join(self.dir_save, f'{num_img:05d}.pth')

        self.dt = {}

    def getPER(self):
        '''
        Inference edge regions with a learned deep neural net
        '''

        edrg = EdgeRegion(args, ni, scaling=True)
        print(f'\tobtaining pre-edge region...')

        input, output = edrg.getEr()
        pre_er = np.where(output > .5, 1., 0.)

        _dt = {'img': input, 'output': output, 'pre_er': pre_er}
        self.dt.update(_dt)
        mts.saveFile(self.dt, self.path_dict)

        self.sts.imshow(input, 'img.pdf')
        self.sts.imshow(output, 'output.pdf', cmap='gray')
        self.sts.imshow(pre_er, 'pre_er.pdf', cmap='gray')

        return 0

    def refinePER(self):
        _dt = mts.loadFile(self.path_dict)
        img = _dt['img']
        pre_er = _dt['pre_er']
        CP = RefinePreER(img, pre_er, self.sts)
        print(f'\trefining pre-edge region...')

        lbl_reg = label(CP.bar_er, background=1, connectivity=1)
        rein = Reinitial(dt=0.1, width=10, tol=0.01)
        phi0 = [rein.getSDF(np.where(lbl_reg == l, -1., 1.)) 
                for l in range(1, np.max(lbl_reg)+1)]
                
        _dt.update({
            'bar_er': CP.bar_er, 'sk': CP.sk, 
            'erfa': CP.erfa, 'gadf': CP.fa, 'phi0': phi0
            })
        mts.saveFile(_dt, self.path_dict)

        self.sts.imcontour(CP.img, phi0, 'phi0.pdf')
        return 0

    def snake(self):
        _dt = mts.loadFile(self.path_dict)
        print(f'\tactive contours...')
        snk = Snake(_dt, self.dir_save)

        snk.dict['phi_res'] = snk.phi_res
        mts.saveFile(snk.dict, self.path_dict)

        self.sts.imshow(snk.erfa, 'erfa.pdf', cmap='gray')
        self.sts.imshow(snk.use_er, 'use_er.pdf', cmap='gray')
        self.sts.imcontour(snk.img, snk.phi_res, 'phi_res.pdf')
        return 0

    def idReg(self):
        _dt = mts.loadFile(self.path_dict)
        print(f'\tindentifying regions...')
        idreg = IdRegion(_dt, self.dir_save)

        mts.saveFile(idreg.dict, self.path_dict)


if __name__=='__main__':
    args = get_args()
    if args.ALL:
        args.pre_er = True
        args.refine = True
        args.snake = True
        args.id_reg = True

    # imgs = args.imgs if args.imgs else [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    # imgs = args.imgs if args.imgs else [4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    # imgs = args.imgs if args.imgs else [0, 1, 5, 8, 17, 18]
    # imgs = args.imgs if args.imgs else [2, 3, 4, 6]
    # imgs = args.imgs if args.imgs else [9, 10, 11, 16]
    # imgs = args.imgs if args.imgs else [13, 14, 20, 21]
    imgs = args.imgs if args.imgs else [32]

    today = time.strftime("%y%m%d", time.localtime(time.time()))
    # label_test = '1'
    label_test = None
    if label_test == None:
        dir_result = join('results', f'er_net/{today}/')
    else:
        dir_result = join('results', f'er_net/{today}_{label_test}/')
    mts.makeDir(dir_result)

    for ni in imgs:
        tseg = TeethSeg(dir_result, ni)

        # Inference pre-edge regions with a learned deep neural net
        if args.pre_er: tseg.getPER()
        if args.refine: tseg.refinePER()
        if args.snake: tseg.snake()
        if args.id_reg: tseg.idReg()
