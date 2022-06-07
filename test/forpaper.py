import os
import numpy as np
import matplotlib.pyplot as plt
import myTools as mts
from reinitial import Reinitial


# dir_save = 'forpaper/'
# sts = mts.SaveTools(dir_save)

# def evolve(phi, img, lbl):
#     # phi3
#     k = 0
#     clr = plt.cm.get_cmap('rainbow', len(phi))
#     _phi = np.copy(phi)
#     while True:
#         kapp, gx, gy, ng = mts.kappa(_phi.transpose((1, 2, 0)))
#         kapp = kapp.transpose((2, 0, 1))
#         ng = ng.transpose((2, 0, 1))

#         reg = np.where(_phi < 1, _phi - 1, 0)
#         all_reg = np.sum(reg, axis=0)
#         Fc = all_reg - reg
#         F = 0.5 * kapp + (- 1 - Fc) * (1 - img) + img
#         _phi += dt * F

#         try:
#             os.mkdir(dir_save + f'Fig8_{lbl}')
#         except:
#             pass
#         if k % 1 == 0:
#             plt.figure(1)
#             plt.clf()
#             plt.imshow(img, 'gray')
#             for kk, ps in enumerate(_phi):
#                 plt.contour(ps, levels=[0], colors=[clr(kk)], linewidths=3)
#             plt.pause(.1)
#             mts.savecfg(dir_save + f'Fig8_{lbl}/iter{k:04d}.png')
#         if k % 1 == 0:
#             _phi = rein.getSDF(_phi)
#         if k > 2000: break

#         k += 1

# rein = Reinitial(dim=2, dim_stack=0)

# sz = 256
# cen = sz / 2 - .5
# dt = .2

# _sz = int(16/9*sz)
# img1 = np.zeros((sz, _sz))
# Y, X = np.mgrid[:sz, :_sz]

# num_seed = 12
# np.random.seed(900320)
# cx = np.random.choice(np.arange(10, _sz-10), num_seed,)
# cy = np.random.choice(np.arange(10, sz-10), num_seed)
# r = np.random.choice(np.arange(10, 70), num_seed)
# init1 = [np.where((X - cx[ii])**2 + (Y - cy[ii])**2 < r[ii]**2, -1., 1.) for ii in range(num_seed)]
# init1 = np.array(init1)
# phi1 = rein.getSDF(init1)

# r1, r2 = 73, 70
# img2 = np.where((X - cen)**2 + (Y - cen)**2 < r1**2, 1., 0.)
# img2 = np.where((X - cen)**2 + (Y - cen)**2 < r2**2, 0., img2)

# i, o = 60, 100
# init2 = []
# init2.append(np.where((X - cen)**2 + (Y - cen)**2 < i**2, -1., 1.))
# init2.append(np.where((X - cen)**2 + (Y - cen)**2 > o**2, -1., 1.))
# init2 = np.array(init2)
# phi2 = rein.getSDF(init2)

# r3, r4 = 63, 60
# ang = 30 * np.pi / 180
# img3 = np.where((X - cen)**2 + (Y - cen)**2 < r3**2, 1., 0.)
# img3 = np.where((X - cen)**2 + (Y - cen)**2 < r4**2, 0., img3)
# img3 = np.where(((Y - cen) - np.tan(ang)*(X - cen) < 0) * ((Y - cen) + np.tan(ang)*(X - cen) > 0), 0., img3)
# img3 = img3.T

# i, o = 50, 120
# init3 = []
# init3.append(np.where((X - cen)**2 + (Y - cen)**2 < i**2, -1., 1.))
# init3.append(np.where((X - cen)**2 + (Y - cen)**2 > o**2, -1., 1.))
# init3 = np.array(init3)
# phi3 = rein.getSDF(init3)

# img4 = plt.imread('/home/users/mireiffe/Documents/Python/TeethSeg/data/images/er1.png')
# img4 = np.where(np.mean(img4, axis=2) < .5, 1., 0.)
# m, n = img4.shape
# Y, X = np.mgrid[0:m, 0:n]

# cy = 212, 300, 318
# cx = 315, 440, 365
# r = 25, 25, 250
# init4 = []
# init4.append(np.where((X - cx[0])**2 + (Y - cy[0])**2 < r[0]**2, -1., 1.))
# init4.append(np.where((X - cx[1])**2 + (Y - cy[1])**2 < r[1]**2, -1., 1.))
# init4.append(np.where((X - cx[2])**2 + (Y - cy[2])**2 > r[2]**2, -1., 1.))
# init4 = np.array(init4)
# phi4 = rein.getSDF(init4)

# evolve(phi1, img1, 'img1')
# # evolve(phi2, img2, 'img2')
# # evolve(phi3, img3, 'img3')
# # evolve(phi4, img4, 'img4')


dir_img = '/home/users/mireiffe/Documents/Python/TeethSeg/data/testimgs/'
num_img = 13

try:
    path_img = dir_img + f'1{num_img:05d}.png'
    img = plt.imread(path_img)
except FileNotFoundError:
    path_img = dir_img + f'1{num_img:05d}.jpg'
    img = plt.imread(path_img)

m, n = img.shape[:2]
# tgr = 100
tgr = 226

plt.figure()
plt.imshow(img)
plt.plot([0, n-1], [100, 100], 'lime', linewidth=2)

R = img[100, :, 0] / 255
G = img[100, :, 1] / 255
B = img[100, :, 2] / 255

xaxis = range(n)
plt.figure()
plt.plot(xaxis, R, 'r')
plt.plot(xaxis, G, 'g')
plt.plot(xaxis, B, 'b')
plt.axis('tight')
plt.grid('on')

plt.figure()
plt.plot(xaxis, G/R, 'black')
plt.plot(xaxis, B, 'b')
plt.axis('tight')
plt.grid('on')

plt.figure()
plt.plot(xaxis, G/R - .3*B, 'black')
plt.axis('tight')
plt.grid('on')

pass