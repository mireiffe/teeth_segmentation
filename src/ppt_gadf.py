from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from gadf import GADF

img = plt.imread('tstimg_inpt.png')[:, :, :3]
# img = Image.open('tstimg_inpt.png')

dt = loadmat('ppt.mat')
I = dt['I']
er = dt['er']
fa = dt['fa']
# gadf = GADF(img, sig=2)
# fa = gadf.Fa
# er = gadf.Er

lv = 2
X, Y = np.indices((int((er.shape[0] + (lv - 1)) // lv), int((er.shape[1] + (lv - 1))//lv)))

plt.figure()
plt.imshow(np.zeros_like(I[::lv, ::lv, :]))
# Q = plt.quiver(Y, X, fa[::lv, ::lv, 1], fa[::lv, ::lv, 0], color='red', units='xy', angles='xy', scale_units='xy', scale=2)
Q = plt.quiver(Y, X, fa[::lv, ::lv, 1], fa[::lv, ::lv, 0], color='red', angles='xy', scale_units='xy', scale=1, lw=20)
plt.show()

xxx = 1