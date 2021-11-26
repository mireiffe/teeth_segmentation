import numpy as np
from time import time
from skimage.measure import label
from reinitial import Reinitial
import matplotlib.pyplot as plt

import myTools as mts



def distrib(img:np.ndarray, reg):
    nc = img.shape[-1]
    if img.ndim == 3:
        img = img.transpose((2, 0, 1))
    mu = img.mean(where=reg)
    var = img.var(where=reg)
    return mu, var

class ThreeRegions():
    jet_alpha = mts.colorMapAlpha(plt)
    def __init__(self, img) -> None:
        self.img = img
        self.m, self.n = self.img.shape[:2]
        self.n_ch = self.img.shape[-1]

    def setting(self, phi):
        self.phi = phi

        self.globalReg()
        self.bandReg()
        self.calParams()

    def globalReg(self):
        self.reg1_i = self.phi < 0
        self.reg1_o = self.phi > 0

    def bandReg(self, gamma=7):
        self.band = np.abs(self.phi) < gamma
        self.reg2_i = self.reg1_i * self.band
        self.reg2_o = self.reg1_o * self.band

    def localReg(self, x, y, delta=25):
        _d = delta // 2
        y1 = np.clip(y - _d, 0, self.m)
        x1 = np.clip(x - _d, 0, self.n)

        y2 = np.clip(y + _d + 1, 0, self.m)
        x2 = np.clip(x + _d + 1, 0, self.n)

        return x1, x2, y1, y2

    def calParams(self):
        self.mu1_i, self.var1_i = distrib(self.img, self.reg1_i)
        self.mu1_o, self.var1_o = distrib(self.img, self.reg1_o)
        self.mu2_i, self.var2_i = distrib(self.img, self.reg2_i)
        self.mu2_o, self.var2_o = distrib(self.img, self.reg2_o)

        self.mu3_i, self.var3_i = np.zeros_like(self.phi), np.zeros_like(self.phi)
        self.mu3_o, self.var3_o = np.zeros_like(self.phi), np.zeros_like(self.phi)
        # idx_band = np.where(self.band)
        
        idx_band = np.where(np.abs(self.phi) < 2)
        for y, x in zip(*idx_band):
            x1, x2, y1, y2 = self.localReg(x, y)
            _img = self.img[y1:y2, x1:x2, ...].transpose((2, 0, 1))
            _reg_i = self.reg2_i[y1:y2, x1:x2]
            _reg_o = self.reg2_o[y1:y2, x1:x2]
            _img_i = _img * _reg_i
            _img_o = _img * _reg_o

            if _reg_i.sum() == 0:
                self.mu3_i[y, x] = 0
                self.var3_i[y, x] = 0
            else:
                self.mu3_i[y, x] = _img_i.mean(where=_reg_i)
                self.var3_i[y, x] = _img_i.var(where=_reg_i)
            if _reg_o.sum() == 0:
                self.mu3_o[y, x] = 0
                self.var3_o[y, x] = 0
            else:
                self.mu3_o[y, x] = _img_o.mean(where=_reg_o)
                self.var3_o[y, x] = _img_o.var(where=_reg_o)

    def force(self):
        def funPDF(X, mu, sig):
            return np.exp(-(X - mu)**2 / 2 / (sig + mts.eps)**2) / np.sqrt(2 * np.pi) / (sig + mts.eps)

        _img = self.img.mean(axis=2)
        # P1_i = funPDF(_img, self.mu1_i, self.var1_i**.5)
        # P1_o = funPDF(_img, self.mu1_o, self.var1_o**.5)
        # F1 = np.sign(np.where(self.band, P1_i - P1_o, 0.))

        # P2_i = funPDF(_img, self.mu2_i, self.var2_i**.5)
        # P2_o = funPDF(_img, self.mu2_o, self.var2_o**.5)
        # F2 = np.sign(np.where(self.band, P2_i - P2_o, 0.))

        P3_i = funPDF(_img, self.mu3_i, self.var3_i**.5)
        P3_o = funPDF(_img, self.mu3_o, self.var3_o**.5)
        F3 = np.sign(np.where(self.band, P3_i - P3_o, 0.))

        # V2 = np.where(np.abs(self.mu3_i - self.mu3_o) < .1, 1, F3)
        # V2 = (np.where(np.abs(self.mu3_i - self.mu3_o) < .001, 1, F3) - 1)
        # V2 = (np.where(np.abs(self.mu3_i - self.mu3_o) < .01, 0, F3) - 1/2)
        V2 = (np.where(np.abs(self.mu3_i - self.mu3_o) < .05, 0, F3))
        # V2 = F3

        return V2
        
        x, y = 341, 179
        # x, y = 301, 81
        # x, y = 244, 167

        for x, y in [(341, 179), (301, 81), (244, 167)]:
            x1, x2, y1, y2 = self.localReg(x, y)
            _img = self.img[y1:y2, x1:x2, ...].transpose((2, 0, 1))
            _reg_i = self.reg2_i[y1:y2, x1:x2]
            _reg_o = self.reg2_o[y1:y2, x1:x2]
            _phi = self.phi[y1:y2, x1:x2]
            _img_i = _img * _reg_i
            _img_o = _img * _reg_o

            plt.figure(); plt.imshow(_img.transpose((1,2,0))); 
            plt.imshow((_img_i+_img_o).transpose((1, 2, 0)), alpha=.7); 
            plt.contour(_phi, levels=[0], colors='red')
            plt.figure(); plt.contour(_phi, levels=[0], colors='red'); plt.imshow(F3[y1:y2, x1:x2], alpha=1)

        self.mu3_i[y, x] = _img_i.sum() / _reg_i.sum() / self.n_ch
        self.var3_i[y, x] = _img_i.var(where=_reg_i)
        self.mu3_o[y, x] = _img_o.sum() / _reg_o.sum() / self.n_ch
        self.var3_o[y, x] = _img_o.var(where=_reg_o)

if __name__ == '__main__':
    dt = mts.loadFile('/home/users/mireiffe/Documents/Python/TeethSeg/results/er_net/210826/00000/00000.pth')

    img = dt['img']
    phi = dt['phi'][..., 0]
    lbl_phi = label(phi < 0)
    reg = (lbl_phi == 2)
    reinit = Reinitial()
    psi = reinit.getSDF(.5 - 1. * reg)
    dt = .2
    teg = ThreeRegions(img)
    teg.setting(psi)

    from celluloid import Camera

    fig = plt.figure(1)
    ax = plt.subplot(111)
    camera = Camera(fig)
    _k = 0
    while True:
        _k += 1
        teg.setting(psi)

        V = mts.gaussfilt(teg.force(), 1)
        psi -= dt * V

        ax.imshow(img[55:210, 215:370, ...])
        ax.contour(psi[55:210, 215:370, ...], levels=[0], colors='red')
        # ax.imshow(V[70:202, 226:361, ...], alpha=.3)
        # plt.show()
        ax.text(0.5, 1.01, f'iteration: {_k}', transform=ax.transAxes)
        # plt.draw()
        # plt.pause(0.1)
        camera.snap()
        # ax.cla()

        print(f'\riterations: {_k}')

        if _k == 14:
            xxx = 1

        # if _k > 90:
        #     animation = camera.animate()
        #     animation.save(
        #         'V2.mp4',
        #         dpi=100,
        #         savefig_kwargs={
        #             'pad_inches': 'tight'
        #         }
        #     )
        #     break

        if _k % 2 == 0:
            psi = reinit.getSDF(psi)
    

    xxx = 1
    
