'''
data loader {img, label}
'''
import os
import logging
import random
from os.path import join, splitext

import cv2
from PIL import Image
import numpy as np
import pickle

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms

# import matplotlib.pyplot as plt


class ErDataset(Dataset):
    '''
    make a list of data
    '''
    def __init__(self, cfg_augs, dir_data, split, wid_dil='auto', mode=None):
        self.cfg_augs = cfg_augs
        self.dir_data = dir_data
        self.mode = mode
        if wid_dil == 'auto':
            self.wid_dil = wid_dil
        else:
            self.wid_dil = int(wid_dil)
        
        self.files = []
        for file in os.listdir(dir_data):
            _num = int(file[:-4])
            if any([_num < splt[1] and _num >= splt[0] for splt in split]):
                self.files += [splitext(file)[0]]

        logging.info(f'Creating dataset with {len(self.files)} examples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # load file
        name_file = self.files[index]
        path_file = join(self.dir_data, name_file +'.pth')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        img = data['img']
        label = data['label_er']

        self.m, self.n = label.shape
        if self.wid_dil == 'auto':
            _wid_dil = round((self.m + self.n) / 2 / 300)
        else:
            _wid_dil = self.wid_dil
        if _wid_dil > 0:
            label = cv2.dilate(label.astype(float), np.ones((2 * _wid_dil + 1, 2 * _wid_dil + 1)), iterations=1)
            
        label = np.where(label > .001, 1., 0.)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        label = label * 255
        label = Image.fromarray(label.astype('uint8'), mode='L')

        if self.mode == 'test':
            input, target = self.transformTest(img, label)
        else:
            input, target = self.transform(img, label)
        return input, target

    def transform(self, img, label):
        # Resize
        rsize = random.randint(256, 512)
        resize = transforms.Resize(size=rsize, interpolation=InterpolationMode.BICUBIC)
        img = resize(img)
        label = resize(label)

        # Random crop
        b, c, h, w = transforms.RandomCrop.get_params(
            img, output_size=(256, 256))
        img = TF.crop(img, b, c, h, w)
        label = TF.crop(label, b, c, h, w)
        
        # Random Gaussian smoothing
        if self.cfg_augs.getboolean('rand_gauss'):
            p_gauss = random.random()
            if p_gauss > .5:
                a_gauss, b_gauss = .25, .75
                sig = random.random() * (b_gauss - a_gauss) + a_gauss
                ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
                img = TF.gaussian_blur(img, ksz, sig)

        # Random contrast adjustment
        if self.cfg_augs.getboolean('rand_contrast'):
            p_ctrst = random.random()
            if p_ctrst > .5:
                a_ctrst, b_ctrst = .75, 1.5
                cf = random.random() * (b_ctrst - a_ctrst) + a_ctrst
                img = TF.adjust_contrast(img, contrast_factor=cf)

        # Random Gamma adjustment
        if self.cfg_augs.getboolean('rand_gamma'):
            p_gamma = random.random()
            if p_gamma > .5:
                a_gam, b_gam = .5, 2
                gam = random.random() * (b_gam - a_gam) + a_gam
                img = TF.adjust_gamma(img, gamma=gam, gain=1)

        # Random Saturation adjustment
        if self.cfg_augs.getboolean('rand_saturation'):
            p_strt = random.random()
            if p_strt > .5:
                a_strt, b_strt = .5, 2
                sf = random.random() * (b_strt - a_strt) + a_strt
                img = TF.adjust_saturation(img, saturation_factor=sf)

        # Random horizontal flipping
        if self.cfg_augs.getboolean('rand_flip'):
            p_hrzn = random.random()
            if p_hrzn > 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)

        # Transform to tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)

        # Random noise
        if self.cfg_augs.getboolean('rand_noise'):
            if random.random() > .5:
                a_noise, b_noise = 0.005, 0.015
                g_sig = random.random() * (b_noise - a_noise) + a_noise
                g_noise = img.clone().normal_(0, g_sig)
                img = img + g_noise

        return img, label

    def transformTest(self, img, label):
        # Resize
        _m, _n = self.m, self.n
        while _m % 32 != 0: _m += 1
        while _n % 32 != 0: _n += 1
        
        dm = _m - self.m
        dn = _n - self.n

        padding = transforms.Pad((dn // 2, dm // 2, dn // 2 + dn % 2, dm // 2 + dm % 2), padding_mode='reflect')
        # padding = transforms.Pad((dn // 2, dm // 2, dn // 2 + dn % 2, dm // 2 + dm % 2))
        img = padding(img)
        label = padding(label)

        # Transform to tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)
        return img, label
