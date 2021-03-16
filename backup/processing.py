from os import stat
import cv2
import numpy as np
from numpy.lib.index_tricks import ndindex
from skimage.measure import label

import matplotlib.pyplot as plt

from anisoDiff import anisodiff
from shockFilter import coherence_filter as shockfilt

class Processing():
    eps = np.finfo(float).eps

    def __init__(self, img, sig=2, epsilon=1, smoothing=0):
        '''
        inputs
        -----
        smoothing: 
            0 for anisotrophic diffusion + shock filter
            1 for gaussian blurring
        '''
        self.img_orig = img
        self.epsilon = epsilon
        self.w, self.h = img.shape[:2]

        if smoothing == 0:
            self.img = np.zeros_like(self.img_orig)
            for i in range(self.img_orig.shape[2]):
                for ii in range(20):
                    self.img[..., i] = anisodiff(self.img_orig[..., i], kappa=3, niter=3, option=2)
                    self.img[..., i] = shockfilt(self.img[..., i], sigma=1, str_sigma=11, blend=.5, iter_n=3)
                self.img[..., i] = anisodiff(self.img[..., i], kappa=3, niter=13, option=2)
        elif smoothing == 1:
            self.img = self.gaussfilt(self.img_orig, sig=sig)
        else:
            raise NotImplementedError('Smoothing option must be in {0, 1}')
        # self.img = self.img_orig

        if len(img.shape) > 2:
            self.c = img.shape[2]
        else:
            self.c = 1

    def setup(self):
        self.gadf()
        self.edgeRegion()
        er1 = self.fineEr(iter=5, coeff=4)
        er2 = self.fineEr(er1, iter=2, coeff=4)
        # res = self.dilation(er2, len=2)
        res = self.fineEr(er2, iter=2, coeff=4)

        res = er1
        return res

    def normalGrad(self, img) -> np.ndarray:
        gx, gy = self.imgrad(img)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        return gx / (ng + self.eps), gy / (ng + self.eps)
    
    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

    @staticmethod
    def imgrad(img) -> np.ndarray:
        # ksize = 1: central, ksize = 3: sobel, ksize = -1:scharr
        gx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
        return gx / 2, gy / 2

    @staticmethod
    def directInterp(img: np.ndarray, direct:tuple or list, mag=1) -> np.ndarray:
        m, n = img.shape[:2]
        y, x = np.indices((m, n))

        x_ = x + mag * direct[0]
        y_ = y + mag * direct[1]

        x_ = np.where(x_ < 0, 0, x_)
        x_ = np.where(x_ > n - 1, n - 1, x_)
        y_ = np.where(y_ < 0, 0, y_)
        y_ = np.where(y_ > m - 1, m - 1, y_)

        x1 = np.floor(x_).astype(int)
        x2 = np.ceil(x_).astype(int)
        y1 = np.floor(y_).astype(int)
        y2 = np.ceil(y_).astype(int)

        I1 = img[y1, x1, ...]
        I2 = img[y1, x2, ...]
        I3 = img[y2, x2, ...]
        I4 = img[y2, x1, ...]

        I14 = (y_ - y1) * I4 + (y2 - y_) * I1
        I23 = (y_ - y1) * I3 + (y2 - y_) * I2

        return (x_ - x1) * I23 +(x2 - x_) * I14

    def structTensor(self):
        gx, gy = self.imgrad(self.img)
        Ei = np.array([[gx * gx, gx * gy], [gy * gx, gy * gy]])
        E = Ei.sum(axis=4).transpose((2, 3, 0, 1))
        return E

    @staticmethod
    def eigvecSort(E:np.ndarray) -> tuple:
        v, Q = np.linalg.eig(E)
        _idx = np.argsort(v, axis=-1)[..., ::-1]
        Q_idx = np.stack((_idx, _idx), axis=2)
        sorted_Q = np.take_along_axis(Q, Q_idx, axis=-1)
        return sorted_Q

    def dux(self, v, mag, h):
        '''
        input
        -----
        v: direction \n
        s: maginitude which is coefficient of v \n
        h: increment for finite differential \n
        '''
        _d = v.transpose((2, 0, 1))
        up = np.array([self.directInterp(self.img[..., i], _d, mag + h) 
            for i in range(self.c)])
        un = np.array([self.directInterp(self.img[..., i], _d, mag - h) 
            for i in range(self.c)])
        res = np.sqrt(np.sum(((up - un) / (2 * h)) ** 2, axis=0))
        return res
        
    def gadf(self) -> None:
        if self.c == 1:
            ngx, ngy = self.normalGrad(self.img)

            Ip = self.directInterp(self.img, (ngx, ngy), self.epsilon)
            In = self.directInterp(self.img, (ngx, ngy), -self.epsilon)

            coeff = np.sign(Ip + In - 2 * self.img)
            self.Fa = np.stack((coeff * ngx, coeff * ngy), axis=2)
        elif self.c == 3:
            h = 1E-02
            E = self.structTensor()
            Q = self.eigvecSort(E)
            v = Q[..., 0]

            num_part = 21
            xp = np.linspace(0, self.epsilon, num_part)
            xn = np.linspace(-self.epsilon, 0, num_part)
            yp, yn = [], []
            for p, n in zip(*[xp, xn]):
                yp.append(self.dux(v, p, h))
                yn.append(self.dux(v, n, h))
            
            lx = np.trapz(yp, dx=1 / 20, axis=0) - np.trapz(yn, dx=1 / 20, axis=0)

            self.Fa = np.sign(lx)[..., None] * v
        else:
            raise NotImplemented('Number of image channels is not 1 or 3.')
        return self.Fa

    def edgeRegion(self) -> None:
        F_ = np.stack((self.directInterp(self.Fa[..., 0], (self.Fa[..., 0], self.Fa[..., 1])),
            self.directInterp(self.Fa[..., 1], (self.Fa[..., 0], self.Fa[..., 1]))), axis=2)
        indic = np.sum(self.Fa * F_, axis=2)
        self.Er = np.where(indic < 0, 1, 0)
        return self.Er

    @staticmethod
    def smallRegion(er, iter=5, coeff=4) -> tuple:
        lbl = label(er, background=0,connectivity=1)
        sz_reg = {}
        for i in range(1, lbl.max() + 1):
            sz_reg[i] = np.sum(lbl == i)
        _lst = list(sz_reg.values())
        _mu = np.mean(_lst)
        _sig = np.std(_lst)

        cnt = 0
        while True:
            cnt += 1
            lim_v = _mu + coeff * _sig
            _items = list(sz_reg.items())
            for k, sr in _items:
                if sr > lim_v: del sz_reg[k]

            if cnt > 3: break
            
            _lst = list(sz_reg.values())
            _mu = np.mean(_lst)
            _sig = np.std(_lst)
        
        part_small = np.zeros_like(lbl)
        for k in sz_reg.keys():
            part_small += (lbl == k)
        part_large = er - part_small
        return part_large, part_small

    def delEr(self, er):
        gimg = self.img_orig.mean(axis=2)
        _lbl = label(er, background=0,connectivity=1)
        
        N = {}
        Sig = {}
        for i in range(1, _lbl.max() + 1):
            N[i] = np.sum(_lbl == i)
            gimg_reg = (_lbl == i) * gimg
            Sig[i] = ((gimg_reg ** 2).sum() - gimg.sum()) / N[i]

        lst_N, lst_Sig = list(N.values()), list(Sig.values())
        mu_N, sig_N = np.mean(lst_N), np.std(lst_N)
        mu_Sig, sig_Sig = np.mean(lst_Sig), np.std(lst_Sig)

        ker = np.ones((3, 3))
        mean_loc = cv2.filter2D(gimg, -1, ker, borderType=cv2.BORDER_REFLECT)
        Sig_loc = np.sqrt(cv2.filter2D((gimg - mean_loc) ** 2, -1, ker, borderType=cv2.BORDER_REFLECT))

        dist_sig = (Sig_loc - mu_Sig) / (sig_Sig + self.eps)

        fun_alpha = lambda x: 1 / (1 + np.exp(-x) + self.eps)
        nx = mu_N + fun_alpha(-dist_sig) * sig_N
        for k, nn in N.items():
            if np.sum((nx > nn) * (_lbl == k)) > .5:
                er = np.where(_lbl == k, 0, er)
        return er

    @staticmethod
    def dilation(er, len=1):
        # _y, _x = np.where(er > .5)
        # _ys = [[_y - 1, _y - 1, _y - 1],
        #         [_y, _y, _y],
        #         [_y + 1, _y +1, _y + 1]]
        # _xs = [[_x - 1, _x, _x + 1],
        #         [_x - 1, _x, _x + 1],
        #         [_x - 1, _x, _x + 1]]
        # er[[_ys, _xs]] = 1

        res = cv2.filter2D(er.astype(np.float), -1, np.ones((2 * len + 1, 2 * len + 1))) > 0
        return res
        
    def fineEr(self, er=None, iter=4, coeff=4):
        if er is not None:
            ler, ser = self.smallRegion(er, iter=iter, coeff=coeff)
        else:
            ler, ser = self.smallRegion(self.Er, iter=iter, coeff=coeff)
        del_s = self.delEr(ser)
        er = ler + del_s
        return er


if __name__ == "__main__":
    import pickle
    with open('data/er_less/00116.pth', 'rb') as f:
        data = pickle.load(f)

    img = data['img']

    proc1 = Processing(img, smoothing=0)
    res1 = proc1.setup()

    proc2 = Processing(img, smoothing=1)
    res2 = proc2.setup()

    plt.figure()
    plt.imshow(proc1.img)
    plt.imshow(res1, alpha=.5, cmap='Reds')
    plt.imshow(res1, alpha=.5, cmap='Greens')
    plt.show()