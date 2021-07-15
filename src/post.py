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
        del_tol = self.m * self.n / 750
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

        indic_kapp = {}
        for ir in range(num_reg):
            if (ir + 1) not in lbl:
                continue
            _reg = np.where(lbl == (ir+1), -1., 1.)
            # _phi = Rein.getSDF(_reg)
            _phi = skfmm.distance(_reg)
            _kapp = self.kappa(_phi, mode=0)[0]
            _kapp = self.gaussfilt(_kapp, sig=2)

            cal_reg = np.abs(_phi) < 2
            p_kapp = np.where(_kapp > 0, _kapp, 0)
            n_kapp = np.where(_kapp < 0, _kapp, 0)

            n_pkapp = ((_kapp > 0) * cal_reg).sum()
            n_nkapp = ((_kapp < 0) * cal_reg).sum()

            if n_pkapp < n_nkapp:
                indic_kapp[ir + 1] = n_pkapp - n_nkapp

            if ir+1 == 18:
                ttt = 1

        # for i, ind in indic_kapp.items():
        #     new_lbl = np.where(new_lbl == i, -1, new_lbl)

        # # second phase
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(self.img.reshape((-1, 3)))
        # kmlbl = kmeans.labels_.reshape((self.m, self.n))

        # km0 = ((kmlbl == 0) * self.img.mean(axis=2)).sum() / (kmlbl == 0).sum()
        # km1 = ((kmlbl == 1) * self.img.mean(axis=2)).sum() / (kmlbl == 1).sum()

        # mustBT = np.argmax([km0, km1])

        # indic_kmeans = {}
        # for ir in range(num_reg):
        #     if (ir + 1) not in lbl:
        #         continue
        #     _reg = np.where(lbl == (ir+1), 1., 0.)
        #     _indic = _reg * kmlbl if mustBT else _reg * (1 - kmlbl)
        #     indic_kmeans[ir+1] = _indic.sum() / _reg.sum()

        temp = lbl
        temp2 = lbl
        new_lbl = lbl
        for i, ind in indic_kapp.items():
            temp = np.where(temp == i, ind, temp)
            temp2 = np.where(temp2 == i, indic_kapp[i], temp2)
            new_lbl = np.where(new_lbl == i, -1, new_lbl)

        plt.figure()
        plt.imshow(temp)
        plt.savefig(f'{self.dir_img}debug_post.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.show()

        return new_lbl

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
        plt.figure()
        plt.imshow(self.lbl0)
        plt.savefig(f'{self.dir_img}lbl0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.lbl)
        plt.savefig(f'{self.dir_img}lbl1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.tot_lbl)
        plt.savefig(f'{self.dir_img}lbl2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.res)
        plt.savefig(f'{self.dir_img}lbl3.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.imshow(self.img)
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        clrs = ['r'] * 100
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors=clrs[i], linewidths=1)
        plt.savefig(f'{self.dir_img}res_0.png', dpi=1024, bbox_inches='tight', facecolor='#eeeeee')
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
        # change alpha values
        color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
        # create a colormap object
        from matplotlib.colors import LinearSegmentedColormap
        map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)
        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)
        plt.close('all')
        plt.figure()
        plt.imshow(self.img)
        _res = np.where(self.res == -1, 0, self.res)
        plt.imshow(_res, alpha=.5, cmap='rainbow_alpha')
        plt.savefig(f'{self.dir_img}res_1.png', dpi=1024, bbox_inches='tight', facecolor='#eeeeee')
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors='r', linewidths=1)
        plt.savefig(f'{self.dir_img}res_2.png', dpi=1024, bbox_inches='tight', facecolor='#eeeeee')

        plt.close('all')
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.lbl0)
        plt.subplot(2, 2, 2)
        plt.imshow(self.tot_lbl)
        plt.subplot(2, 2, 3)
        plt.imshow(self.res)
        plt.subplot(2, 2, 4)
        plt.imshow(self.img)
        # clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        clrs = ['r'] * 100
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors=clrs[i])
        plt.savefig(f'{self.dir_img}res_tot.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.pause(10)


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
