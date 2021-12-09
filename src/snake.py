import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique

from skimage.measure import label

from reinitial import Reinitial
from reinst import ThreeRegions

import myTools as mts


class Snake():
    eps = np.finfo(float).eps
    
    def __init__(self, dict, dir_img) -> None:
        self.dir_img = dir_img
        self.dict = dict
        self.img:np.ndarray = dict['img']
        self.bar_er:np.ndarray = dict['bar_er']
        self.phi0 = dict['phi0']
        self.fa = dict['gadf']
        self.erfa = dict['erfa']

        self.m, self.n = self.bar_er.shape
        self.lbl_er =  label(self.erfa, background=0, connectivity=1)
        self.phi_res = self.snake()

    def snake(self):
        self.phi0 = self.phi0[2:3]
        # self.phi0 = [self.phi0[i] for i in [2, 0, 1, 3, 9]]
        # self.img = self.img[85:155, 110:163, ...]
        # self.bar_er = self.bar_er[85:155, 110:163]
        # self.erfa = self.erfa[85:155, 110:163]
        # self.fa = self.fa[85:155, 110:163, ...]
        
        dt = 0.2
        mu = .5
        n_phis = len(self.phi0)
        cmap = plt.cm.get_cmap('gist_ncar', n_phis)

        # Rein = Reinitial(dt=.2, width=4, tol=0.01, dim_stack=0, fmm=True)
        Rein = Reinitial(dt=.2, width=4, tol=0.01, dim_stack=0)
        teg = [ThreeRegions(self.img) for nph in range(n_phis)]

        # phis = np.copy(self.phi0)[..., 85:155, 110:163]
        phis = np.copy(self.phi0)

        stop_reg = np.ones_like(self.bar_er)
        stop_reg[2:-2, 2:-2] = 0
        
        # oma = cv2.dilate(self.erfa * self.bar_er, kernel=np.ones((3, 3)), iterations=1)
        self.use_er = self.erfa * self.bar_er
        oma = self.use_er
        omc = (1 - oma) * (1 - stop_reg)
        oms = (self.bar_er - oma) * (1 - stop_reg) 

        k = 0
        while True:
            k += 1
            if k % 3 == 0:
                phis = Rein.getSDF(np.where(phis < 0, -1., 1.))
                # phis = Rein.getSDF(phis)

            dist = 1
            regs = np.where(phis < dist, phis - dist, 0)
            all_regs = regs.sum(axis=0)
            Fc = (- (all_regs - regs) - 1)

            for i in range(n_phis):
                teg[i].setting(phis[i])

            gx, gy = mts.imgrad(phis.transpose((1, 2, 0)))
            Fa = - (gx.transpose((2, 0, 1)) * self.fa[..., 1] + gy.transpose((2, 0, 1)) * self.fa[..., 0])
            _Fb = np.array([- tg.force() for tg in teg])
            Fb = mts.gaussfilt((_Fb).transpose((1, 2, 0)), 1)[np.newaxis, ...]
            # Fb = mts.gaussfilt((_Fb).transpose((1, 2, 0)), 1).transpose((2, 0, 1))

            kap = mts.kappa(phis.transpose((1, 2, 0)))[0].transpose((2, 0, 1))
            # F = (Fa + 5*mu*kap)*oma + (Fc + mu*kap)*omc
            F = Fa*oma + Fb*oms + Fc*omc + mu*kap
            new_phis = phis + dt * F

            err = np.abs(new_phis - phis).sum() / new_phis.size
            if err < 1E-04 or k > 200:
            # if err < 1E-04 or k > 1:
                break
        
            if k in [1, 2] or k % 1 == 0:
                plt.figure(1)
                plt.cla()
                # plt.imshow(self.img)
                # plt.imshow(1 - self.bar_er, 'gray', vmin=-1)
                plt.imshow(1 - self.bar_er[85:155, 110:163], 'gray', vmin=-1)
                # plt.imshow((Fb + Fc).sum(axis=0)[85:155, 110:163], vmin=-0.3, cmap=mts.colorMapAlpha(plt, _cmap='rainbow'))
                plt.imshow(oma[85:155, 110:163], cmap=mts.colorMapAlpha(plt, _cmap='gnuplot'), vmax=1.1)
                for i, ph in enumerate(new_phis):
                    _pr = np.where(ph > 0)
                    if len(_pr[0]) == self.m * self.n:
                        continue
                    if i == 0:
                        plt.contour(ph[85:155, 110:163], levels=[0], colors='lime', linewidths=3.5)
                    else:
                        plt.contour(ph[85:155, 110:163], levels=[0], colors='red', linewidths=3.5)
                # plt.title(f'iter = {k:d}')
                # plt.show()
                mts.savecfg(f'/home/users/mireiffe/Documents/Python/TeethSeg/forpaper/ksiam_evolve_ex/ex{k}')
                plt.pause(.1)

                # plt.figure(2)
                # plt.cla()
                # plt.imshow(self.img, 'gray')
                # plt.imshow((Fb*oms).sum(axis=0), alpha=.5)
                # for i, ph in enumerate(new_phis):
                #     _pr = np.where(ph > 0)
                #     if len(_pr[0]) == self.m * self.n:
                #         continue
                #     plt.contour(ph, levels=[0], colors='lime', linewidths=3.5)
                # plt.pause(.1)

            phis = new_phis

        return new_phis
