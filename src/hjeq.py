import sys
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
    cdt2 = (X - _c[0])**2 + (Y - _c[1])**2 >= (_r - 1)**2
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
    # er = skeletonize(er)
    # er = cv2.dilate(np.where(er > .5, 1., 0.), np.ones((2,2)), iterations=1)

    # er = plt.imread('/home/users/mireiffe/Documents/Python/TeethSeg/data/images/er1.png').mean(axis=-1)
    # er = np.where(er < .5, 1., 0.)

    er = np.zeros((5, 5))
    er[2, :2] = 1

    dir_save = '/home/users/mireiffe/Documents/Python/TeethSeg/results'
    _dir = join(dir_save, 'test')
    def hvsd(x, eps=.1):
        return .5 * (1 + 2 / np.pi * np.arctan(x / eps))

    rein = Reinitial(debug=True)
    # __rein = Reinitial(debug=False, _rr=True)
    COH = Coherent(sig=1, rho=10)

    m, n = er.shape
    vlim = [(53, 73), (73, 93)]
    vlim = [(0, n), (0, m)]

    # __psi = __rein.getSDF(2 * (.5 - er)[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]])
    # psi = rein.getSDF(2 * (.5 - er)[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]], _r=__psi)
    psi = rein.getSDF(2 * (.5 - er)[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]])

    sys.exit(0)

    gx, gy = COH.imgrad(psi)
    ng = np.sqrt(gx ** 2 + gy ** 2)
    sk_psi = np.where((ng < .75) * (psi > 0), 1., 0.)


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()



    vpsi = psi[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
    vng = ng[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
    ver = er[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
    
    plt.figure()
    # plt.imshow(vng)
    plt.imshow(.5 - ver)
    # plt.contour(vpsi, levels=[0], colors='red', linestyles='dotted')
    _r = np.abs(vpsi) <3
    for i in range(vlim[1][1] - vlim[1][0]):
        for j in range(vlim[0][1] - vlim[0][0]):
            if _r[i, j]:
                _v = np.round(vpsi[i, j], 1)
                plt.text(j, i, _v, fontsize=8, color='red', ha='center', va='center', clip_on=True)

    # plt.show()

    dt = .3
    iter = 1000
    texton = False
    texton = True
 
    psi1 = psi
    psi2 = psi
    for k in range(iter):    
        kapp1, _, _, ng1 = COH.kappa(psi1, mode=1)
        kapp2, _, _, ng2 = COH.kappa(psi2, mode=1)

        gk1 = COH.gaussfilt(kapp1, sig=.1)
        # gk2 = COH.gaussfilt(kapp2, sig=.5)
        # gk1 = kapp1
        gk2 = kapp2

        # f1 = np.maximum(kapp1 - 1, 0)
        f1 = np.where(psi1 < -.5, 0., np.maximum(gk1 - 1, 0))
        f2 = np.where(psi2 < -.5, 0., np.maximum(gk2 - 1, 0))

        gf1 = COH.gaussfilt(f1, sig=2) * (1 - sk_psi)
        gf2 = COH.gaussfilt(f2, sig=2) * (1 - sk_psi)
        # gf1 = f1
        # gf2 = f2

        vpsi1 = psi1[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vpsi2 = psi2[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vkapp1 = kapp1[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vkapp2 = kapp2[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vgk1 = gk1[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vgk2 = gk2[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vf1 = gf1[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]
        vf2 = gf2[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]

        ax1.cla()
        # ax1.contour(vpsi, levels=[0], colors='red', linestyles='dotted')
        # ax1.contour(vpsi1, levels=[0], colors='red')
        # ax1.imshow(vgk1)
        # ax1.set_title(f'iter = {k}')
        # _r = np.abs(vpsi1) < 2
        # if texton:
        #     for i in range(vlim[1][1] - vlim[1][0]):
        #         for j in range(vlim[0][1] - vlim[0][0]):
        #             if _r[i, j]:
        #                 _v = np.round(vgk1[i, j], 1)
        #                 ax1.text(j, i, _v, fontsize=8, color='black', ha='center', va='center', clip_on=True)
        ax1.imshow(ver, 'gray')

        ax2.cla()
        ax2.contour(vpsi, levels=[0], colors='red', linestyles='dotted')
        ax2.contour(vpsi2, levels=[0], colors='red')
        ax2.imshow(vgk2)
        ax2.set_title(f'iter = {k}')
        _r = np.abs(vpsi2) < 2
        if texton:
            for i in range(vlim[1][1] - vlim[1][0]):
                for j in range(vlim[0][1] - vlim[0][0]):
                    if _r[i, j]:
                        _v = np.round(vgk2[i, j], 1)
                        ax2.text(j, i, _v, fontsize=8, color='black', ha='center', va='center', clip_on=True)

        # plt.show()
        plt.pause(.1)
        # plt.savefig(join(_dir, f"test{k:04d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        # plt.figure(); plt.imshow(ng2[vlim[1][0]:vlim[1][1], vlim[0][0]:vlim[0][1]]); plt.show()

        # psi1 += - dt * f1 * ng1
        psi2 += - dt * f2 * ng2
        

        if k % 5 == 0 or np.sum(np.abs(f2)) < 1E-05:
            psi1 = rein.getSDF(psi1) 
            psi2 = rein.getSDF(psi2) 
            # psi1 = rein.getSDF(.5 - (psi1 < 0)) 
            # psi2 = rein.getSDF(.5 - (psi2 < 0))
