from os.path import join

import pickle
import yaml

import numpy as np

# custom libs
from reinitial import Reinitial


class Initials():
    eps = np.finfo(float).eps
    dir_img = '/home/users/mireiffe/Documents/Python/TeethSeg/data/er_less'
    dir_cfg = 'cfg/info_initial.yml'

    def __init__(self, num_img, radii='auto'):
        self.num_img = num_img
        self.radii = radii

        self.img = self.loadFile(join(self.dir_img, f"{self.num_img:05d}.pth"))['img']
        self.inits = np.transpose(self.getInitials(), (1, 2, 0))
        self.num_inits = self.inits.shape[-1]

        reinit = Reinitial(self.inits, dt=.1, width=5, tol=.01, iter=None, dim=2)
        self.phis = reinit.getSDF()

    @staticmethod
    def loadFile(path: str):
        with open(path, 'rb') as f:
            _d = pickle.load(f)
        return _d

    @staticmethod
    def getSeeds(num_img, dir_cfg):
        with open(dir_cfg) as f:
            _i = yaml.load(f, Loader=yaml.FullLoader)
        return _i[f"T{num_img:05d}"]

    def getInitials(self):
        seeds = self.getSeeds(self.num_img, self.dir_cfg)

        seed_teeth = seeds['teeth']
        seed_gums = seeds['gums']

        y, x = self.img.shape[:2]
        Y, X = np.indices([y, x])

        if self.radii=='auto':
            self.radii = (y + x) // 150

        _init = [np.where((X-sd[0])**2 + (Y-sd[1])**2 < self.radii**2, -1, 1) for sd in seed_teeth]
        _init.append(sum([np.where((X-sd[0])**2 + (Y-sd[1])**2 < self.radii**2, -1, 1) for sd in seed_gums]))
        return _init

if __name__=="__main__":
    import matplotlib.pyplot as plt

    GI = Initials(num_img=51, radii='auto')
    plt.figure()
    plt.imshow(GI.img)

    cmaps = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i in range(GI.num_inits):
        if i < 6:
            cm = cmaps[i]
        else:
            cm = f"#{np.random.randint(0, 255):02X}{np.random.randint(0, 255):02X}{np.random.randint(0, 255):02X}"
        plt.contour(GI.phis[..., i], levels=[0], colors=cm)
    plt.show()
