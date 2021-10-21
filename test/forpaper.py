import numpy as np
import matplotlib.pyplot as plt
import myTools as mts
from reinitial import Reinitial

dir_save = 'forpaper'
sts = mts.SaveTools(dir_save)

rein = Reinitial(dim=2)

sz = 256
cen = sz / 2 - .5
dt = .2

img1 = np.zeros((sz, sz))

r1, r2 = 73, 70
Y, X = np.mgrid[:sz, :sz]
img2 = np.where((X - cen)**2 + (Y - cen)**2 < r1**2, 1., 0.)
img2 = np.where((X - cen)**2 + (Y - cen)**2 < r2**2, 0., img2)

i, o = 60, 90
init2 = []
init2.append(np.where((X - cen)**2 + (Y - cen)**2 < i**2, -1., 1.))
init2.append(np.where((X - cen)**2 + (Y - cen)**2 > o**2, -1., 1.))
init2 = np.transpose(np.array(init2), (1, 2, 0))
phi2 = rein.getSDF(init2).transpose((2, 0, 1))

r3, r4 = 73, 70
ang = 30 * np.pi / 180
Y, X = np.mgrid[:sz, :sz]
img3 = np.where((X - cen)**2 + (Y - cen)**2 < r3**2, 1., 0.)
img3 = np.where((X - cen)**2 + (Y - cen)**2 < r4**2, 0., img3)
img3 = np.where(((Y - cen) - np.tan(ang)*(X - cen) < 0) * ((Y - cen) + np.tan(ang)*(X - cen) > 0), 0., img3)
img3 = img3.T

i, o = 60, 90
init3 = []
init3.append(np.where((X - cen)**2 + (Y - cen)**2 < i**2, -1., 1.))
init3.append(np.where((X - cen)**2 + (Y - cen)**2 > o**2, -1., 1.))
init3 = np.transpose(np.array(init3), (1, 2, 0))
phi3 = rein.getSDF(init3).transpose((2, 0, 1))


# phi3
k = 0
clr = ['r', 'g', 'b', 'y']
_phi3 = np.copy(phi3)
while True:
    kapp, gx, gy, ng = mts.kappa(_phi3.transpose((1, 2, 0)))
    kapp = kapp.transpose((2, 0, 1))
    ng = ng.transpose((2, 0, 1))

    reg = np.where(_phi3 < 1, phi3-1, 0)
    all_reg = np.sum(reg, axis=0)
    Fc = all_reg - reg
    # _phi3 += dt * ( 0.1 * kapp + (- 1 - Fc) * (1 - img3) ) * ng 
    _phi3 += dt * ( 0.1 * kapp + (- 1 - Fc) * (1 - img3) ) * ng 

    if k % 5 == 0:
        plt.figure(1)
        plt.clf()
        plt.imshow(img3, 'gray')
        for kk, ps in enumerate(_phi3):
            plt.contour(ps, levels=[0], colors=clr[kk], linewidth=1.5)
        plt.pause(.5)
    if k % 1 == 0:
        _phi3 = rein.getSDF(_phi3)
    if k > 100: break

    k += 1