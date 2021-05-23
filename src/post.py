import os 

import pickle
import numpy as np
import matplotlib.pyplot as plt

def loadFile(path):
    with open(path, 'rb') as f:
        _dt = pickle.load(f)
    return _dt


path_dt = 'results/test_lvset00051/dict01001.pck'

data = loadFile(path_dt)

er = data['er']
phi = data['phi']

pre_res = data
