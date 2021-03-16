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
    dir_cfg = 'cfg/info_initial.yml'
    dir_img = '/home/users/mireiffe/Documents/Python/TeethSeg/data/er_less'

    def __init__(self, num_img:int, er:np.ndarray, radii='auto', dt:float=0.1):
        self.num_img = num_img
        self.er = er
        self._er = np.expand_dims(np.where(er > .5, 1., 0.), axis=2)
        self.radii = radii
        self.dt = dt

        self.img = self.loadFile(join(self.dir_img, f"{self.num_img:05d}.pth"))['img']
        proc = Processing(self.img)
        self.fa = proc.gadf()

        inits = np.transpose(self.getInitials(), (1, 2, 0))

        self.reinit = Reinitial(dt=.1, width=5, tol=.01, iter=None, dim=2)
        self.phis0 = self.reinit.getSDF(inits)

    @staticmethod
    def loadFile(path: str):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return _d

    def getInitials(self) -> list:
        # get initial seeds
        with open(self.dir_cfg) as f:
            seeds = yaml.load(f, Loader=yaml.FullLoader)[f"T{self.num_img:05d}"]
        seed_teeth = seeds['teeth']
        seed_gums = seeds['gums']

        y, x = self.img.shape[:2]
        Y, X = np.indices([y, x])

        if self.radii=='auto':
            self.radii = (y + x) // 150

        _init = []
        _init_1 = sum([np.where((X-sd[0])**2 + (Y-sd[1])**2 < self.radii**2, 1, 0) 
                for _i, sd in enumerate(seed_teeth) if _i % 2 == 0])
        _init.append(-_init_1 + .5)
        _init_2 = sum([np.where((X-sd[0])**2 + (Y-sd[1])**2 < self.radii**2, 1, 0) 
                for _i, sd in enumerate(seed_teeth) if _i % 2 != 0])
        _init.append(-_init_2 + .5)
        _gums = sum([np.where((X-sd[0])**2 + (Y-sd[1])**2 < self.radii**2, 1, 0) for sd in seed_gums])
        _init.append(-_gums + .5)
        return _init

    # ballooon inflating cores
    def force_balloon(self, phis, mu):
        _kp = self.kappa(phis)
        if np.ndim(_kp) < 3:
            _kp = np.expand_dims(_kp, axis=2)
        _f = mu * _kp - (4 * (.5 - self._er))
        return _f

    def force_gadf(self, phis):
        gx, gy = self.imgrad(phis)
        if np.ndim(gx) < 3:
            gx = np.expand_dims(gx, axis=2)
            gy = np.expand_dims(gy, axis=2)
        _f = - self._er * (self.fa[..., 0:1] * gx + self.fa[..., 1:2] * gy)
        return _f

    def update(self, phis, mu=.01):
        if np.ndim(phis) < 3:
            phis = np.expand_dims(phis, axis=2)
        fb = self.force_balloon(phis, mu)
        fa = self.force_gadf(phis)
        _phis = phis + self.dt * fb
        return _phis

    def kappa(self, phis, ksz=1, h=1, mode=0):
        x, y = self.imgrad(phis)
        if mode == 0:
            ng = np.sqrt(x**2 + y**2 + self.eps)
            nx, ny = x / ng, y / ng
            xx, _ = self.imgrad(nx)
            _, yy = self.imgrad(ny)
            return xx + yy
        elif mode == 1:
            xx, yy, xy = self.imgrad(phis, order=2)
            res_den = xx * y * y - 2 * x * y * xy + yy * x * x
            res_num = np.power(x ** 2 + y ** 2, 1.5)
            return res_den / (res_num + self.eps)

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
            _img = np.pad(img, (1, 1), mode='symmetric')

            gxx = img[:, 2:, ...] + img[:, :-2, ...] - 2 * img[:, 1:-1, ...]
            gyy = img[2:, :, ...] + img[:-2, :, ...] - 2 * img[1:-1, :, ...]
            gxy = img[2:, 2:, ...] + img[:-2, :-2, ...] - img[2:, :-2, ...] - img[:-2, 2:, ...]
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
