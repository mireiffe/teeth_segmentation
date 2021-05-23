import os
import numpy as np


num_img = [56, 57, 58, 59]

for ni in num_img:
    dir = f'results/test_lvset000{ni}/'

    flst = os.listdir(dir)

    for ff in flst:
        if ff[-3:] != 'png':
            continue
        fnum = int(ff[4:9])

        if fnum % 3 == 1:
            nfnum = (fnum // 3) + 1

            os.system(f'mv {dir}test{fnum:05d}.png {dir}ntest{nfnum:05d}.png')

