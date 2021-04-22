from os.path import join

from skimage.morphology import skeletonize
import numpy as np
import cv2
import matplotlib.pyplot as plt

from shockFilter import coherence_filter
from reinitial import Reinitial


class Coherent():
    eps = np.finfo(float).eps
    def __init__(self, sig, rho):
        self.sig = sig
        self.rho = rho

    def getCoOri(self, img):
        img_orig = img
        img = self.gaussfilt(img, sig=2)
        E = self.structTensor(img)

        eig_E = self.eigSort(E)
        coh = eig_E[1][..., 1]

        sz = img_orig.shape
        Y, X = np.indices((sz[0], sz[1]))

        plt.figure()
        plt.imshow(img_orig, 'gray')
        plt.quiver(X, Y, coh[..., 0], coh[..., 1], angles='xy', scale_units='xy', scale=None, color='blue')
        plt.show()

        return coh

    def structTensor(self, img):
        gx, gy = self.imgrad(img)
        Ei = np.array([[gx * gx, gx * gy], [gy * gx, gy * gy]])
        if Ei.ndim == 5:
            E = Ei.sum(axis=4).transpose((2, 3, 0, 1))
        else:
            E = Ei.transpose((2, 3, 0, 1))
        return E

    @staticmethod
    def eigSort(E:np.ndarray) -> tuple:
        v, Q = np.linalg.eig(E)
        _idx = np.argsort(v, axis=-1)[..., ::-1]
        Q_idx = np.stack((_idx, _idx), axis=2)
        sorted_v = np.take_along_axis(v, _idx, axis=-1)
        sorted_Q = np.take_along_axis(Q, Q_idx, axis=-1)
        return sorted_v, sorted_Q 

    @staticmethod
    def imgrad(img, order=1) -> np.ndarray:
        # ksize = 1: central, ksize = 3: sobel, ksize = -1:scharr
        if order == 1:
            gx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
            gy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
            return gx / 2, gy / 2
        elif order == 2:
            gxx = cv2.Sobel(img, -1, 2, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
            gxy = cv2.Sobel(img, -1, 1, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
            gyy = cv2.Sobel(img, -1, 0, 2, ksize=1, borderType=cv2.BORDER_REFLECT)
            return gxx / 1, gyy / 1, gxy / 4

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

    def kappa(self, phis, ksz=1, h=1, mode=0, eps=1):
        x, y = self.imgrad(phis)
        ng = np.sqrt(x**2 + y**2 + self.eps)
        if mode == 0:
            ng = 1
            nx, ny = x / ng, y / ng
            xx, _ = self.imgrad(nx)
            _, yy = self.imgrad(ny)
            return xx + yy, x, y, ng
        elif mode == 1:
            xx, yy, xy = self.imgrad(phis, order=2)
            res_den = xx * y * y - 2 * x * y * xy + yy * x * x
            res_num = np.power(x ** 2 + y ** 2, 1.5)
            return res_den / (res_num + self.eps), x, y, ng


if __name__ == '__main__':
    _sz = 128, 128
    _c = _sz[0] // 2, _sz[1] // 2
    # _r = (_sz[0] + _sz[1]) // 20
    _r = 20
    ang = 10 * np.pi / 180        # degree

    [X, Y] = np.indices((_sz[0], _sz[1]))
    cdt1 = (X - _c[0])**2 + (Y - _c[1])**2 < _r**2
    cdt2 = (X - _c[0])**2 + (Y - _c[1])**2 >= (_r - 2)**2
    er = np.where(cdt1 * cdt2, 1., 0.)
    er = np.where(((Y - _c[1]) - np.tan(ang) * (X - _c[0]) < 0) * ((Y - _c[1]) + np.tan(ang) * (X - _c[0])) > 0, 0., er)

    er_ = np.zeros_like(er)
    for x in zip(range(61, 67), range(61, 67)):
        er_[x[0], x[1]] = 1
    for x in zip(range(62, 68), range(61, 67)):
        er_[x[0], x[1]] = 1
    for x in zip(range(63, 69), range(61, 67)):
        er_[x[0], x[1]] = 1
    
    
    er = er + er_

    # er = plt.imread('/home/users/mireiffe/Documents/Python/TeethSeg/data/images/er1.png').mean(axis=-1)
    # er = np.where(er < .5, 1., 0.)

    dir_save = '/home/users/mireiffe/Documents/Python/TeethSeg/results'
    _dir = join(dir_save, 'test')
    def hvsd(x, eps=.1):
        return .5 * (1 + 2 / np.pi * np.arctan(x / eps))

    rein = Reinitial()
    COH = Coherent(sig=1, rho=10)

    psi = rein.getSDF(.5 - er)

    kapp1, gx, gy, ng = COH.kappa(psi, mode=0)
    kapp2, _, _, _ = COH.kappa(psi, mode=1)


    m, n = kapp1.shape
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.contour(psi, levels=[0], colors='red')
    ax1.imshow(kapp1, vmin=-1.7, vmax=1.7)
    _r = np.abs(psi) < 2
    for i in range(m):
        for j in range(n):
            if _r[i, j]:
                _v = np.round(kapp1[i, j], 2)
                ax1.text(j, i, _v, fontsize=8, color='black', ha='center', va='center', clip_on=True)

    ax2.contour(psi, levels=[0], colors='red')
    ax2.imshow(kapp2, vmin=-1.7, vmax=1.7)
    _r = np.abs(psi) < 2
    for i in range(m):
        for j in range(n):
            if _r[i, j]:
                _v = np.round(kapp2[i, j], 2)
                ax2.text(j, i, _v, fontsize=8, color='black', ha='center', va='center', clip_on=True)
    plt.show()
