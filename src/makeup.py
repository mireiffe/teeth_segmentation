# system libs
from os.path import join

# general libs
import argparse
import numpy as np
import matplotlib.pyplot as plt

# custom libs
import myTools as mts
from teethSeg import PseudoER, InitContour, Snake, IdRegion

# global variables
jet_alpha = mts.colorMapAlpha(plt)
brg_alpha = mts.colorMapAlpha(plt, cmap='brg')

def get_args():
    parser = argparse.ArgumentParser(description='Individual tooth segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-I", "--imgs", dest="imgs", nargs='+', type=int, default=[],
                             required=False, metavar="NI", 
                             help="indices of images")
    parser.add_argument("-d", "--device", dest="device", nargs='+', type=str, default=0,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("-p", "--pseudo_er", dest="pseudo_er",
                             required=False, action='store_true',
                             help="Network inference for making pseudo edge region")
    parser.add_argument("-c", "--init_contours", dest="inits",
                             required=False, action='store_true',
                             help="make initial contours")
    parser.add_argument("-s", "--snake", dest="snake",
                             required=False, action='store_true',
                             help="snake")
    parser.add_argument("-i", "--id_region", dest="id_region",
                             required=False, action='store_true',
                             help="Identification of regions")
    parser.add_argument("-A", "--ALL", dest="ALL",
                             required=False, action='store_true',
                             help="Do every process in a row")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default='cfg/inference.ini',
                             required=False, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()
    # return parser.parse_args(args=[])


class TeethSeg():
    def __init__(self, dir_result, num_img, sts:mts.SaveTools, args) -> None:
        self.num_img = num_img
        self.sts = sts
        self.args = args
        
        print(f'image {num_img}, initiating...')

        # set an empty dictionary
        self.path_dict = join(dir_result, f'{num_img:05d}.pth')

        try:
            self._dt:dict = mts.loadFile(self.path_dict)
        except FileNotFoundError:
            self._dt = {}

    def pseudoER(self):
        '''
        Inference pseudo edge-regions with a deep neural net
        '''
        print(f'\tobtaining pre-edge region...')

        PER = PseudoER(self.args, self.num_img, scaling=True)
        inpt, oupt = PER.getEr()             # network inference
        per0 = np.where(oupt > .5, 1., 0.)     # pseudo edge-region

        self._dt.update({
            'img': inpt, 'output': oupt, 'per0': per0
        })
        mts.saveFile(self._dt, self.path_dict)

        # saving images
        self.sts.imshow(inpt, 'img.png')
        self.sts.imshow(oupt, 'output.png', cmap='gray')
        self.sts.imshow(per0, 'pseudo_er0.png', cmap='gray')

        return 0

    def initContour(self):
        print(f'\trefining pre-edge region...')
        img, per0 = self._dt['img'], self._dt['per0']

        initC = InitContour(img, per0)

        self._dt.update({
            'phi0': initC.phi0, 'per': initC.per,
            })
        mts.saveFile(self._dt, self.path_dict)

        self.sts.imcontour(img, initC.phi0, 'phi0_img.pdf')
        self.sts.imcontour(initC.per, initC.phi0, 'phi0_per.pdf', cmap='gray')

        return 0

    def snake(self):
        print(f'\tactive contours...')
        img, per, phi0 = self._dt['img'], self._dt['per'], self._dt['phi0']

        SNK = Snake(img, per, phi0)
        phi_res = SNK.snake()
        
        self._dt.update({
            'phi_res': phi_res, 'gadf': SNK.fa, 
            'er': SNK.er,
        })
        mts.saveFile(self._dt, self.path_dict)

        self.sts.imshow(SNK.er, 'er.png', cmap='gray')
        self.sts.imshow(per * SNK.er, 'use_er.png', cmap='gray')
        self.sts.imshows([per, SNK.er], 'er_per.png', cmap=['gray', jet_alpha], alphas=[None, None])
        self.sts.imcontour(img, phi_res, 'phires_img.pdf')
        return 0

    def idReg(self):
        print(f'\tindentifying regions...')
        img, phi_res = self._dt['img'], self._dt['phi_res']

        ir = IdRegion(img, phi_res)

        self._dt.update({'lbl_reg': ir.lbl_reg, 'res': ir.res})
        mts.saveFile(self._dt, self.path_dict)

        self.sts.imshow(ir.lbl_reg, 'lbl_reg.png')
        self.sts.imshow(ir.res, 'res.png')
        self._showSaveMax(self.img, 'res_c.pdf', contour=self.res)
