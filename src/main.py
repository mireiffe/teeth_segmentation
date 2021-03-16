from os.path import join
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# custom libs
from edge_region import EdgeRegion
from balloon import Balloon


tol = 0.01
dir_save = '/home/users/mireiffe/Documents/Python/TeethSeg/results'

def get_args():
    parser = argparse.ArgumentParser(description='Balloon inflated segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img", dest="num_img", type=int, default=False,
                             required=True, metavar="NI", 
                             help="number of image")
    parser.add_argument("--device", dest="device", nargs='+', type=str, default=False,
                             required=False, metavar="DVC",
                             help="name of dataset to use")
    parser.add_argument("--cfg", dest="path_cfg", type=str, default=False,
                             required=True, metavar="CFG", 
                             help="configuration file")
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()

    # get edge regions from network
    edrg = EdgeRegion(args.path_cfg, args.num_img)
    er = edrg.getEr()

    bln = Balloon(args.num_img, er, radii='auto', dt=0.3)

    # # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    phis = bln.phis0

    fig, ax = bln.setFigure(phis)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    _k = 0
    while True:
        _k += 1
        _vis = _k % 10 == 0
        _save = _k % 5 == 0
        _reinit = _k % 1 == 0

        new_phis = bln.update(phis)
        print(f"\riteration: {_k}", end='')

        if _save or _vis:
            bln.drawContours(_k, phis, ax)
            if _save: plt.savefig(join(dir_save, *['test3', f"test{_k:04d}.png"]))
            if _vis: plt.pause(1)
        
        err = np.abs(new_phis - phis).sum() / np.ones_like(phis).sum()
        if err < tol:
            break

        if _reinit:
            new_phis = bln.reinit.getSDF(new_phis)
        phis = new_phis

