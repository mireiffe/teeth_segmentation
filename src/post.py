from os.path import join

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label


class PostProc():
    def __init__(self, dir_img) -> None:
        self.dir_img = dir_img

        path_seg = join(f'{dir_img}dict.pck')
        _dt = self.loadFile(path_seg)
        self.img = _dt['img']
        self.er = _dt['er']
        self.phi = _dt['phis'][..., 0]
        self.m, self.n = self.er.shape

        self.labeling()
        self.zeroReg()
        self.distSize()
        self._saveSteps()

    def labeling(self):
        '''
        labeling connected region (0 value for not assigned region)
        '''
        seg_res = np.where(self.phi < 0, 1., 0.)
        self.lbl = label(seg_res, background=0, connectivity=1)
        del_tol = self.m * self.n / 1000
        for lbl_i in range(1, np.max(self.lbl) + 1):
            idx_i = np.where(self.lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                self.lbl[idx_i] = 0

    def zeroReg(self):
        '''
        Assing 0 regions by using intensity values
        '''
        idx_zero_reg = np.where(self.lbl == 0)
        _lbl = np.zeros_like(self.lbl)
        for idx in zip(*idx_zero_reg):
            dumy = np.zeros_like(self.lbl).astype(float)
            dumy[idx] = 1.
            while True:
                dumy = np.where(cv2.filter2D(dumy, -1, np.ones((3,3))) > .01, 1., 0.)
                _idx = np.where(dumy * (self.lbl != 0) > 0)
                if len(_idx[0]) > 0:
                    _lbl[idx[0], idx[1]] = self.lbl[_idx[0][0], _idx[1][0]]
                    break
        self.lbl2 = self.lbl + _lbl

    def distSize(self):
        '''
        distribution for size of region
        '''
        num_reg = np.max(self.lbl2)
        sz_reg = [np.sum(self.lbl2 == (i + 1)) for i in range(num_reg)]

        self.mu_sz = sum(sz_reg) / num_reg
        mu_sz_2 = sum([sr ** 2 for sr in sz_reg]) / num_reg
        self.sig_sz = np.sqrt(mu_sz_2 - self.mu_sz ** 2)

    def _saveSteps(self):
        res = self.lbl2
        
        cv2.imwrite(f'{self.dir_img}lbl1.png', self.lbl, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        cv2.imwrite(f'{self.dir_img}lbl2.png', self.lbl2, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        # cv2.imwrite(f'{self.dir_img}lbl3.png', self.lbl, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        # cv2.imwrite(f'{self.dir_img}lbl1.png', self.lbl, params=[cv2.IMWRITE_PNG_COMPRESSION,0])

        # plt.figure()
        # plt.imshow(self.img)
        # clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        # for i in range(np.max(res)):
        #     plt.contour(np.where(res == i, -1., 1.), levels=[0], colors=clrs[i])
        # plt.savefig(f'{self.dir_img}lbl3_c.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        plt.close('all')
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(self.lbl)
        plt.subplot(2,2,2)
        plt.imshow(self.lbl2)
        plt.subplot(2,2,3)
        plt.imshow(res)
        plt.subplot(2,2,4)
        plt.imshow(self.img)
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        for i in range(np.max(res)):
            plt.contour(np.where(res == (i + 1), -1., 1.), levels=[0], colors=clrs[i])
        plt.savefig(f'{self.dir_img}lbl3_all.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.show()

    @staticmethod
    def loadFile(path):
        with open(path, 'rb') as f:
            _dt = pickle.load(f)
        return _dt
