import os
from os.path import join

import cv2
import pickle
import numpy as np

import matplotlib.pyplot as plt
from numpy.lib.function_base import iterable

eps = np.finfo(float).eps

def saveFile(dict:dict, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path:str) -> None:
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def makeDir(path) -> None:
    try:
        os.mkdir(path)
        print(f"Created a directory {path}")
    except OSError:
        pass

def grad(v:np.ndarray, dim:int, method:str):
    '''
    Output
    ---
    d = dy, dx, dz
    '''
    if method in {'forward', 'backward'}:
        h = 1
    elif method in {'central'}:
        h = 2
    dx = v[:, h:, ...] - v[:, :-h, ...]
    dy = v[h:, :, ...] - v[:-h, :, ...]
    if dim == 2:
        return dy / h, dx / h
    elif dim == 3:
        dz = v[:, :, h:] - v[:, :, :-h]
        return dy / h, dx / h, dz / h

def imgrad(v:np.ndarray, dim:int=2, order=1, method='central'):
    '''
    Output
    ---
    d = dy, dx, dz if order==1,
        dyy, dxx, dxy if order==2.
    '''
    if order == 1:
        _d = grad(v, dim, method)
        d = []
        for i in range(dim):
            sz = list(v.shape)
            sz[i] = 1
            _z = np.zeros(sz)
            if method in {'forward'}:
                d.append(np.concatenate((_d[i], _z), axis=i))
            elif method in {'backward'}:
                d.append(np.concatenate((_z, _d[i]), axis=i))
            elif method in {'central'}:
                d.append(np.concatenate((_z, _d[i], _z), axis=i))
    elif order == 2:
        _v = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode='symmetric')
        dxx = _v[1:-1, 2:, ...] + _v[1:-1, :-2, ...] - 2 * _v[1:-1, 1:-1, ...]
        dyy = _v[2:, 1:-1, ...] + _v[:-2, 1:-1, ...] - 2 * _v[1:-1, 1:-1, ...]
        dxy = _v[2:, 2:, ...] + _v[:-2, :-2, ...] - _v[2:, :-2, ...] - _v[:-2, 2:, ...]
    return dyy, dxx, dxy

def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
    if ksz is None:
        ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
    return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

def kappa(phis, ksz=1, h=1, mode=0):
    y, x = imgrad(phis, dim=2, order=1, method='central')
    if mode == 0:
        ng = np.sqrt(x**2 + y**2 + eps)
        nx, ny = x / ng, y / ng
        _, xx = imgrad(nx)
        yy, _ = imgrad(ny)
        res = xx + yy
    elif mode == 1:
        yy, xx, xy = imgrad(phis, dim=2, order=2, method='central')
        den = xx * y * y - 2 * x * y * xy + yy * x * x
        num = np.power(x ** 2 + y ** 2, 1.5)
        ng = np.sqrt(x**2 + y**2 + eps)        # just for output
        res = den / (num + eps)
    return res, x, y, ng

def cker(rad):
    rad = np.maximum(round(rad), 1)
    Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
    cen_pat = rad
    return np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)


class SaveTools():
    def __init__(self, dir_save) -> None:
        self.dir_save = dir_save

    def imshow(self, img, name_save, cmap=None):
        fig = plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.savefig(join(self.dir_save, name_save), dpi=512, bbox_inches='tight', facecolor='#eeeeee')
        plt.close(fig)

    def imshows(self, imgs: iterable, name_save: str, cmaps: iterable, alphas: iterable):
        fig = plt.figure()
        for im, cm, alph in zip(*[imgs, cmaps, alphas]):
            plt.imshow(im, cmap=cm, alpha=alph)
        plt.savefig(join(self.dir_save, name_save), dpi=512, bbox_inches='tight', facecolor='#eeeeee')
        plt.close(fig)


def colorMapAlpha(_plt, _cmap='jet', name='jet_alpha') -> None:
    # get colormap
    ncolors = 256
    color_array = _plt.get_cmap(_cmap)(range(ncolors))
    # change alpha values
    color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
    # create a colormap object
    from matplotlib.colors import LinearSegmentedColormap
    map_object = LinearSegmentedColormap.from_list(name=name, colors=color_array)
    # register this new colormap with matplotlib
    _plt.register_cmap(cmap=map_object)
    return name


if __name__ == '__main__':
    pass