from os.path import join

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label
import skfmm
from sklearn.cluster import KMeans

from reinitial import Reinitial


class PostProc():
    eps = np.finfo(float).eps
    
    def __init__(self, dict, dir_img) -> None:
        self.dir_img = dir_img

        self.img = dict['img']
        self.seg_er = dict['seg_er']
        self.er = dict['er']
        self.phi = dict['phi'][..., 0]
        self.m, self.n = self.er.shape

        self.lbl0 = self.labeling()
        self.soaking()
        self.lbl = self.labeling()
        self.tot_lbl = self.zeroReg(self.lbl)
        # self.distSize()
        self.res = self.regClass(self.tot_lbl)
        self._saveSteps()

    def labeling(self):
        '''
        labeling connected region (0 value for not assigned region)
        '''
        seg_res = np.where(self.phi < 0, 1., 0.)
        lbl = label(seg_res, background=0, connectivity=1)
        del_tol = self.m * self.n / 1000
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                lbl[idx_i] = 0

        return lbl

    def soaking(self):
        self.phi = np.where((self.phi > 0) * (self.seg_er < .5), -1, self.phi)

    def zeroReg(self, lbl):
        '''
        Assing 0 regions by using intensity values
        '''
        idx_zero_reg = np.where(lbl == 0)
        _lbl = np.zeros_like(lbl)
        for idx in zip(*idx_zero_reg):
            dumy = np.zeros_like(lbl).astype(float)
            dumy[idx] = 1.
            while True:
                dumy = np.where(cv2.filter2D(dumy, -1, np.ones((3,3))) > .01, 1., 0.)
                _idx = np.where(dumy * (lbl != 0) > 0)
                if len(_idx[0]) > 0:
                    _lbl[idx[0], idx[1]] = lbl[_idx[0][0], _idx[1][0]]
                    break
        return lbl + _lbl

    def regClass(self, lbl):
        num_reg = lbl.max()
        Rein = Reinitial(dt=.3, width=5, tol=.05, iter=None, dim=2)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(self.img.reshape((-1, 3)))

        indic = []
        for ir in range(num_reg):
            _reg = np.where(lbl == (ir+1), -1., 1.)
            _phi = Rein.getSDF(_reg)
            # _phi = skfmm.distance(_reg)
            _kapp = self.kappa(_phi, mode=0)[0]
            _kapp = self.gaussfilt(_kapp, sig=2)

            # plt.figure()
            # plt.imshow(_kapp)
            # plt.imshow(np.where(_kapp > 0, 1., -1.), cmap='jet', alpha=.5)
            # plt.show()

            # plt.figure()
            # plt.hist(_kapp.flatten(), bins=256, range=(_kapp.min(), _kapp.max()), log=True, histtype='step')
            # plt.show()

            cal_reg = np.abs(_phi) < 3
            # cal_reg = 1
            kapp_p = np.where((_kapp > 0), 1., 0) * cal_reg
            kapp_n = (1 - kapp_p) * cal_reg
            # sz_pos = kapp_p.sum()
            # sz_neg = (1 - kapp_p).sum()
            # sz_pos = _kapp.sum()
            # sz_neg = (1 - kapp_p).sum()

            # indic.append(np.sum(_kapp * (np.abs(_phi) < 5)))
            # indic.append(np.sum(_kapp))

            indic.append((kapp_p.sum() - kapp_n.sum()))

        temp = lbl
        for i, ind in enumerate(indic):
            temp = np.where(temp == (i+1), ind, temp)
            if ind < 300:
                lbl = np.where(lbl == (i+1), -1, lbl)

        plt.figure()
        plt.imshow(temp)
        plt.show()

    def distSize(self):
        '''
        distribution for size of region
        '''
        num_reg = np.max(self.tot_lbl)
        sz_reg = [np.sum(self.tot_lbl == (i + 1)) for i in range(num_reg)]

        self.mu_sz = sum(sz_reg) / num_reg
        mu_sz_2 = sum([sr ** 2 for sr in sz_reg]) / num_reg
        self.sig_sz = np.sqrt(mu_sz_2 - self.mu_sz ** 2)

    def _saveSteps(self):
        res = self.lbl2
        
        cv2.imwrite(f'{self.dir_img}lbl1.png', self.lbl0 / self.lbl0.max() * 255, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        cv2.imwrite(f'{self.dir_img}lbl2.png', self.lbl / self.lbl.max() * 255, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        cv2.imwrite(f'{self.dir_img}lbl3.png', self.lbl2 / self.lbl2.max() * 255, params=[cv2.IMWRITE_PNG_COMPRESSION,0])

        plt.figure()
        plt.imshow(self.lbl0)
        plt.savefig(f'{self.dir_img}lbl0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.lbl)
        plt.savefig(f'{self.dir_img}lbl1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.lbl2)
        plt.savefig(f'{self.dir_img}lbl2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.img)
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        for i in range(np.max(res)):
            plt.contour(np.where(res == i, -1., 1.), levels=[0], colors=clrs[i])
        plt.savefig(f'{self.dir_img}res.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

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

    def kappa(self, phis, ksz=1, h=1, mode=0):
        x, y = self.imgrad(phis)
        if mode == 0:
            ng = np.sqrt(x**2 + y**2 + self.eps)
            nx, ny = x / ng, y / ng
            xx, _ = self.imgrad(nx)
            _, yy = self.imgrad(ny)
            return xx + yy, x, y, ng
        elif mode == 1:
            xx, yy, xy = self.imgrad(phis, order=2)
            res_den = xx * y * y - 2 * x * y * xy + yy * x * x
            res_num = np.power(x ** 2 + y ** 2, 1.5)
            ng = np.sqrt(x**2 + y**2 + self.eps)        # just for output
            return res_den / (res_num + self.eps), x, y, ng

    @staticmethod
    def imgrad(img: np.ndarray, order=1, h=1) -> np.ndarray:
        '''
        central difference
        '''
        nd = img.ndim
        if nd < 3:
            img = np.expand_dims(img, axis=-1)
        if order == 1:
            _x_ = img[:, 2:, ...] - img[:, :-2, ...]
            x_ = img[:, 1:2, ...] - img[:, :1, ...]
            _x = img[:, -1:, ...] - img[:, -2:-1, ...]

            _y_ = img[2:, :, ...] - img[:-2, :, ...]
            y_ = img[1:2, :, ...] - img[:1, :, ...]
            _y = img[-1:, :, ...] - img[-2:-1, :, ...]

            gx = np.concatenate((x_, _x_, _x), axis=1)
            gy = np.concatenate((y_, _y_, _y), axis=0)
            if nd < 3:
                gx = gx[..., 0]
                gy = gy[..., 0]
            return gx / (2 * h), gy / (2 * h)
        elif order == 2:
            _img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='symmetric')

            gxx = _img[1:-1, 2:, ...] + _img[1:-1, :-2, ...] - 2 * _img[1:-1, 1:-1, ...]
            gyy = _img[2:, 1:-1, ...] + _img[:-2, 1:-1, ...] - 2 * _img[1:-1, 1:-1, ...]
            gxy = _img[2:, 2:, ...] + _img[:-2, :-2, ...] - _img[2:, :-2, ...] - _img[:-2, 2:, ...]
            if nd < 3:
                gxx = gxx[..., 0]
                gyy = gyy[..., 0]
                gxy = gxy[..., 0]
            return gxx / (h * h), gyy / (h * h), gxy / (4 * h * h)

    @staticmethod
    def loadFile(path):
        with open(path, 'rb') as f:
            _dt = pickle.load(f)
        return _dt

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)