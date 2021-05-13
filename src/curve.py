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

    def kappa(self, phis, ksz=1, h=1, mode=0, eps=1):
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


if __name__ == '__main__':
    _sz = 128, 128
    _c = _sz[0] // 2, _sz[1] // 2
    # _r = (_sz[0] + _sz[1]) // 20
    _r = 20
    ang = 20 * np.pi / 180        # degree

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

    for kk in range(3):
        psi = rein.getSDF(.5 - er)


        gx, gy = COH.imgrad(psi)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        ec = np.where((ng < .75) * (er > .5), 1., 0.)
        ek = skeletonize(ec)

        k = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)

        gap = 3
        num_pts = 10
        len_cv = gap * (num_pts - 1) + num_pts
        iter = 10
        _l = 10

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        while True:
                
            # find end points
            ind_ek = np.where(ek > .5)
            end_ek = np.zeros_like(ek)
            ind_cv = []
            for iy, ix in zip(*ind_ek):
                if ek[iy - 1:iy + 2, ix - 1:ix + 2].sum() < 3:
                    ind_cv.append([(iy, ix)])
                    end_ek[iy, ix] = 1

            # find curves
            for idx in ind_cv:
                py, px = idx[0]
                for iy, ix in idx:
                    ptch = ek[iy - 1:iy + 2, ix - 1:ix + 2]
                    _ind = np.where(ptch > .5)
                    for _iy, _ix in zip(*_ind):
                        _pre = (_iy + iy - 1 != py) or (_ix + ix - 1!= px)
                        _curr = (_iy != 1) or (_ix != 1)
                        if _pre and _curr:
                            idx.append((iy + _iy - 1, ix + _ix - 1))
                    py, px = iy, ix
                    if len(idx) >= len_cv:
                        break
                
            vec_cv = []
            Tx, Ty = np.zeros_like(er), np.zeros_like(er)
            new_er = np.zeros_like(er)
            for idx in ind_cv:
                if len(idx) < len_cv - 1:
                    vec_cv.append((0, 0))
                    continue
                b = np.array(list(zip(*idx[::gap + 1]))).T
                _D = np.arange(num_pts)
                D = np.array([_D * _D, _D, np.ones_like(_D)]).T
                abc = np.linalg.lstsq(D, b)[0]

                ss = 1 / gap / 5
                pt = np.arange(0, -1, -ss)

                yy = np.round(abc[0, 0] * pt * pt + abc[1, 0] * pt + abc[2, 0]).astype(int)
                xx = np.round(abc[0, 1] * pt * pt + abc[1, 1] * pt + abc[2, 1]).astype(int)

                for ii, (_yy, _xx) in enumerate(list(zip(yy, xx))[1:]):
                    if psi[yy[ii], xx[ii]] < psi[_yy, _xx]:
                        new_er[_yy, _xx] = 1
                
                
            #     _vy, _vx = T1[0] - TT[0] / 2, T1[1] - TT[1] / 2
            #     _nv = np.sqrt(_vx ** 2 + _vy ** 2)
            #     vy, vx = _vy / (_nv + 0.0001), _vx / (_nv + 0.0001)

            #     vec_cv.append((vy, vx))

            # new_er = np.zeros_like(er)
            # for ii, idx in enumerate(ind_cv):
            #     iy, ix = idx[0]
            #     Ty[iy, ix] = vec_cv[ii][0]
            #     Tx[iy, ix] = vec_cv[ii][1]
            #     for iii in range(1, _l + 1):
            #         _yy, _xx = int(np.round(iy + (iii - 1) * vec_cv[ii][0])), int(np.round(ix + (iii - 1) * vec_cv[ii][1]))
            #         yy, xx = int(np.round(iy + (iii) * vec_cv[ii][0])), int(np.round(ix + (iii) * vec_cv[ii][1]))
            #         if psi[yy, xx] > psi[_yy, _xx] + .1:
            #             new_er[int(np.round(iy + iii * vec_cv[ii][0])), int(np.round(ix + iii * vec_cv[ii][1]))] = 1
            #             break

            if k > iter:
                break
            else:
                k += 1
                ek = np.where(new_er + ek > .5, 1., 0.)
                ek = skeletonize(ek)
        
            [Y, X] = np.indices(_sz)
            ax.cla()
            ax.imshow(ek, 'gray')
            for idx in ind_cv:
                _y, _x = list(zip(*idx[::gap]))
                ax.plot(_x, _y, 'r.-')
            plt.imshow(new_er, alpha=.5)
            plt.pause(.5)
            # plt.show()
            # plt.savefig(join(_dir, f"test{k + kk*iter:04d}.png"), dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        er = cv2.dilate(np.where(ek > .5, 1., 0.), np.ones((3,3)), iterations=1)
