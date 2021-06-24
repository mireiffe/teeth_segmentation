import os
from os.path import join

from skimage.morphology import skeletonize
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

from shockFilter import coherence_filter
from skimage.measure import label

from reinitial import Reinitial


class CurveProlong():
    gap = 3
    num_pts = 10
    maxlen_cv = gap * (num_pts - 1) + num_pts

    def __init__(self, er, img):
        # er[151:153, 150:152] = 0
        
        self.er0 = er
        self.er = er
        self.img = img
        self.m, self.n = self.er.shape

        self.preSet()

    def preSet(self):
        self.removeHoles()
        self.skeletonize()
        self.endPoints()
        plt.figure()
        plt.imshow(self.er, 'gray')
        self.dilation(wid=2)

        self.removeHoles()
        self.skeletonize()
        plt.figure()
        plt.imshow(self.sk, 'gray')
        plt.show()
        self.endPoints()
        self.findCurves()

    def reSet(self):
        self.sk = skeletonize(self.sk)
        self.endPoints()
        self.findCurves()

    def dilation(self, wid=2):
        # # Dilation by norm of gradient
        # _img = self.gaussfilt(self.img.mean(axis=2), sig=2)
        # gx, gy = self.imgrad(_img)
        # ng = np.sqrt(gx ** 2 + gy ** 2)
        # ker = np.ones((2*wid + 1, 2*wid + 1))
        # mu = ng.sum() / np.where(ng > 0, 1, 0).sum()
        # _er = np.where(ng <= mu * .1, self.er, 0.)
        # _er = cv2.dilate(_er, ker, iterations=1)

        _pre = np.zeros_like(self.er)
        for idx in self.ind_end:
            _pre[idx[0]] = 1

        rad = 8
        Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
        cen_pat = rad
        ker = np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)
        _pre = cv2.filter2D(_pre, -1, kernel=ker, borderType=cv2.BORDER_REFLECT)

        # _ends = np.where(_pre == 1, self.er, 0)
        # dil_ends = cv2.dilate(_ends, np.ones((3, 3)), iterations=2)

        plt.imshow(self.sk, alpha=.5)
        plt.imshow(_pre, 'Reds', alpha=.5)
        plt.show()

        dil_ends = _pre

        self.er = np.where(dil_ends + self.er > .5, 1., 0.)

    def removeHoles(self, param_sz=1000):
        lbl = label(self.er, background=1, connectivity=1)
        del_tol = self.m * self.n / param_sz
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                self.er[idx_i] = 1

    def skeletonize(self):
        '''
        skeletonization of edge region
        '''
        rein = Reinitial()
        self.psi = rein.getSDF(.5 - self.er)
        gx, gy = self.imgrad(self.psi)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        
        self.sk = np.where((ng < .80) * (self.er > .5), 1., 0.)
        self.sk = skeletonize(self.sk)

    def endPoints(self):
        # find end points
        _ind = np.where(self.sk > .5)
        _end = np.zeros_like(self.sk)
        self.ind_end = []
        for iy, ix in zip(*_ind):
            _ptch = self.sk[iy - 1:iy + 2, ix - 1:ix + 2]
            if (_ptch.sum() < 3) and (_ptch.sum() > 0):
                self.ind_end.append([(iy, ix)])
                _end[iy, ix] = 1

    def findCurves(self):
        # find curves
        for idx in self.ind_end:
            y0, x0 = idx[0]
            for y_i, x_i in idx:
                ptch = self.sk[y_i-1:y_i+2, x_i-1:x_i+2]
                ind_ptch = np.where(ptch > .5)
                # if len(ind_ptch[0]) > 3: 
                    # continue
                for yy_i, xx_i in zip(*ind_ptch):
                    _pre = (yy_i + y_i - 1 != y0) or (xx_i + x_i - 1!= x0)
                    _curr = (yy_i != 1) or (xx_i != 1)
                    if _pre and _curr:
                        idx.append((y_i + yy_i - 1, x_i + xx_i - 1))
                y0, x0 = y_i, x_i
                if len(idx) >= self.maxlen_cv:
                    break

        lst_del = []
        for idx in self.ind_end:
            if len(idx) < self.maxlen_cv // 3:
                lst_del.append(idx)
        for ld in lst_del: self.ind_end.remove(ld)

    def dilCurve(self):
        _gap = 1 / self.gap / 5
        pts = np.arange(0, -self.gap, -_gap)
        self.new_er = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        
        _D = np.arange(self.num_pts)
        D = np.array([_D * _D, _D, np.ones_like(_D)]).T
        for k, pt in enumerate(pts[1:]):
            for iii, idx in enumerate(self.ind_end):
                if (len(idx) < self.maxlen_cv - 1) or banned[iii]:
                    continue

                b = np.array(list(zip(*idx[::self.gap + 1]))).T
                abc = np.linalg.lstsq(D, b, rcond=None)[0]
                abc[-1, :] = b[0]

                yy = np.round(abc[0, 0] * pt * pt + abc[1, 0] * pt + abc[2, 0]).astype(int)
                xx = np.round(abc[0, 1] * pt * pt + abc[1, 1] * pt + abc[2, 1]).astype(int)

                _yy = np.round(abc[0, 0] * pts[k] * pts[k] + abc[1, 0] * pts[k] + abc[2, 0]).astype(int)
                _xx = np.round(abc[0, 1] * pts[k] * pts[k] + abc[1, 1] * pts[k] + abc[2, 1]).astype(int)

                if (yy == _yy) and (xx == _xx):
                    continue

                if yy >= self.m or xx >= self.n:
                    banned[iii] = 1
                    continue
                elif self.psi[_yy, _xx] > self.psi[yy, xx]:
                # elif _ng[yy, xx] < .75 and _psi[yy, xx] > 0:
                    banned[iii] = 1
                    continue
                # elif ek[yy, xx] > .5 and psi[yy, xx] > 0:
                #     banned[iii] = 1
                #     continue
                self.new_er[yy, xx] = 1
                    
            if np.abs(self.new_er).sum() > 0:
                self.sk = np.where(self.new_er + self.sk > .5, 1., 0.)
                
    @staticmethod
    def imgrad(img) -> np.ndarray:
        # ksize = 1: central, ksize = 3: sobel, ksize = -1:scharr
        gx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
        return gx / 2, gy / 2

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)


if __name__ == '__main__':
    pass