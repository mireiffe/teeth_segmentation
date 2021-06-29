from os import error
from os.path import join
import yaml
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

# custom libs
from reinitial import Reinitial
from processing import Processing


class Balloon():
    eps = np.finfo(float).eps
    rad_init = 7

    def __init__(self, er:np.ndarray, wid:int, radii='auto', dt:float=0.1):
        self.er = er
        self._er = np.expand_dims(np.where(er > .5, 1., 0.), axis=2)
        self.radii = radii
        self.dt = dt
        self.wid = wid
        
        if wid % 2 == 0:
            raise NameError('wid must be an odd number')

        # get initial SDF
        inits = self.getInitials()
        self.reinit = Reinitial(dt=.1, width=None, tol=.01, iter=None, dim=2)
        self.phis0 = self.reinit.getSDF(inits)

        self.psi = self.reinit.getSDF(np.where(self.er > .5, -1., 1.))

        self.grid_Y, self.grid_X = np.indices(self._er.shape[:2])
        self._erx, self._ery = self.imgrad(self.reinit.getSDF(.5 - self._er))
        self._erng = np.sqrt(self._erx**2 + self._ery**2)

    @staticmethod
    def loadFile(path: str):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return _d

    def getInitials(self) -> list:
        # get initial seeds
        rad = self.rad_init
        Y, X = np.indices([3 * rad, 3 * rad])
        cen_pat = 1.5 * rad - .5
        pat = np.where((X - cen_pat)**2 + (Y - cen_pat)**2 < rad**2, -1., 1.)

        y, x = self.er.shape[:2]
        py, px = y - 3 * rad, x - 3 * rad

        gap = 2
        _init = np.pad(pat, ((py // gap, py - py // gap), (px // gap, px - px // gap)), mode='symmetric')
        _init = np.expand_dims(_init, axis=-1)
        return np.where(self._er < .5, _init, 1.)

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

    # ballooon inflating cores
    def force(self, phis, gx, gy, ng, skltn=.8):
        ker_E = np.ones((3 * self.wid + 1, 3 * self.wid + 1))
        ker_R = np.ones((self.wid, self.wid))

        _T = np.where(np.abs(gx * self._erx + gy * self._ery) < skltn, 1., 0.)[..., 0]
        _E = cv2.dilate(self._er, ker_E, iterations=1)
        _R = cv2.dilate(np.where((ng < skltn) * (phis > 0), 1., 0.), ker_R, iterations=1)

        _f = np.where(_R * _E * _T, 2., -1.)
        g_f = self.gaussfilt(_f, sig=.5)
        return np.expand_dims(g_f, axis=-1)

    def update(self, phis, mu=10):
        if np.ndim(phis) < 3:
            phis = np.expand_dims(phis, axis=2)
        kp, gx, gy, ng = self.kappa(phis, mode=0)
        fb = self.force(phis, gx, gy, ng)
        
        _e = np.expand_dims(self.gaussfilt(self.er, sig=.1), axis=-1)
        _e = _e / _e.max()

        kp = kp[..., 0]
        kp = kp / np.abs(kp).max()
        kp = np.where((kp > .9) + (kp < -.01), kp, 0)
        _f = np.expand_dims(mu * kp - self.psi + 1.5*self.er, axis=-1)
        _phis = phis + self.dt * _f
        return _phis

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
    def imgrad(img, order=1, h=1) -> np.ndarray:
        '''
        central difference
        '''
        if order == 1:
            _x_ = img[:, 2:, ...] - img[:, :-2, ...]
            x_ = img[:, 1:2, ...] - img[:, :1, ...]
            _x = img[:, -1:, ...] - img[:, -2:-1, ...]

            _y_ = img[2:, :, ...] - img[:-2, :, ...]
            y_ = img[1:2, :, ...] - img[:1, :, ...]
            _y = img[-1:, :, ...] - img[-2:-1, :, ...]

            gx = np.concatenate((x_, _x_, _x), axis=1)
            gy = np.concatenate((y_, _y_, _y), axis=0)
            return gx / (2 * h), gy / (2 * h)
        elif order == 2:
            _img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='symmetric')

            gxx = _img[1:-1, 2:, ...] + _img[1:-1, :-2, ...] - 2 * _img[1:-1, 1:-1, ...]
            gyy = _img[2:, 1:-1, ...] + _img[:-2, 1:-1, ...] - 2 * _img[1:-1, 1:-1, ...]
            gxy = _img[2:, 2:, ...] + _img[:-2, :-2, ...] - _img[2:, :-2, ...] - _img[:-2, 2:, ...]
            return gxx / (h * h), gyy / (h * h), gxy / (4 * h * h)

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

    # visualizations
    def setFigure(self, phis):
        if np.ndim(phis) < 3:
            phis = np.expand_dims(phis, axis=2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.er > .5, 'gray')

        cmaps = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for i in range(phis.shape[-1]):
            if i < 6:
                cm = cmaps[i]
            else:
                np.random.seed(900501 + i)
                rcm = np.random.randint(0, 255, 3)
                cm = f"#{rcm[0]:02X}{rcm[0]:02X}{rcm[0]:02X}"
            ax.contour(phis[..., i], levels=[0], colors=cm, linestyles='dotted')
            ax.set_title(f"k = {0}")
        return fig, ax

    def drawContours(self, state:int, phis, axis,  mode='rec'):
        cmaps = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        
        # if contour exists, remove it
        if hasattr(self,'contours'):
            for contour in self.contours:
                for coll in contour.collections:
                    coll.remove()
        self.contours = []
        for i in range(phis.shape[-1]):
            if i < 6:
                cm = cmaps[i]
            else:
                np.random.seed(900501 + i)
                rcm = np.random.randint(0, 255, 3)
                cm = f"#{rcm[0]:02X}{rcm[0]:02X}{rcm[0]:02X}"
            self.contours.append(axis.contour(phis[..., i], levels=[0], colors=cm))
            axis.set_title(f"k = {state}")
        return self.contours

    @staticmethod
    def _show(wandtoshow, contour=None):
        plt.figure()
        plt.imshow(wandtoshow)
        plt.grid(which='minor', axis='both', linestyle='-')
        if contour is not None:
            plt.contour(contour, levels=[0], colors='red')
        plt.show()

    def _showVF(self, phis):
        plt.figure()
        plt.imshow(self._er, 'gray')
        plt.contour(phis[..., 0], levels=[0], colors='red')

        x, y = self.imgrad(phis)
        plt.quiver(self.grid_X, self.grid_Y, x[..., 0], y[..., 0], angles='xy', scale_units='xy', scale=None, color='blue')
        plt.quiver(self.grid_X, self.grid_Y, self._erx[..., 0], self._ery[..., 0], angles='xy', scale_units='xy', scale=None, color='red')

        plt.show()
