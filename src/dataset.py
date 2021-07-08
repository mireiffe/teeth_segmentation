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
            st = splitext(file)
            _num = int(st[0])
            if any([_num < splt[1] + 100000 and _num >= splt[0] + 100000 for splt in split]):
                self.files += [st[0] + st[1]]

        logging.info(f'Creating dataset with {len(self.files)} examples')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # load file
        name_file = self.files[index]
        path_file = join(self.dir_data, name_file)

        img = Image.open(path_file).convert('RGB')

        self.n, self.m = img.size

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        input = self.transformTest(img)
        return input,

    def transformTest(self, img):
        # Resize
        _m, _n = self.m, self.n
        while _m % 32 != 0: _m += 1
        while _n % 32 != 0: _n += 1
        
        dm = _m - self.m
        dn = _n - self.n

        padding = transforms.Pad((dn // 2, dm // 2, dn // 2 + dn % 2, dm // 2 + dm % 2), padding_mode='reflect')
        # padding = transforms.Pad((dn // 2, dm // 2, dn // 2 + dn % 2, dm // 2 + dm % 2))
        img = padding(img)

        # Transform to tensor
        img = TF.to_tensor(img)
        return img
