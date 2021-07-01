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
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms


class ErDataset(Dataset):
    '''
    make a list of data
    '''
    def __init__(self, dir_data, split, wid_dil='auto'):
        self.dir_data = dir_data
        
        self.files = []
        for file in os.listdir(dir_data):
            _num = int(file[:-4])
            if _num == split:
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
            
        label = np.where(label > .001, 1., 0.)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        label = label * 255
        label = Image.fromarray(label.astype('uint8'), mode='L')

        input, target = self.transform(img, label)
        return input, target

    def transform(self, img, label):
        # Resize
        _m, _n = self.m, self.n
        while _m % 32 != 0: _m += 1
        while _n % 32 != 0: _n += 1
        
        dm = _m - self.m
        dn = _n - self.n

        padding = transforms.Pad((dn // 2, dm // 2, dn // 2 + dn % 2, dm // 2 + dm % 2))
        img = padding(img)
        label = padding(label)

        # Transform to tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)
        return img, label
