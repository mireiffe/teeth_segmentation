import os
from os.path import join

import pickle
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt
from numpy.lib.function_base import iterable

def saveFile(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)
    return 0

def loadFile(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f)
    return dict

def makeDir(path):
    try:
        os.mkdir(path)
        print(f"Created a directory {path}")
    except OSError:
        pass


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


class ColorMapAlpha():
    def __init__(self, _plt) -> None:
        self.name = 'rainbow_alpha'
        # get colormap
        ncolors = 256
        color_array = _plt.get_cmap('jet')(range(ncolors))
        # change alpha values
        color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
        # create a colormap object
        from matplotlib.colors import LinearSegmentedColormap
        map_object = LinearSegmentedColormap.from_list(name=self.name, colors=color_array)
        # register this new colormap with matplotlib
        _plt.register_cmap(cmap=map_object)
