# system libs
import time
from os.path import join

# custom libs
import myTools as mts
from makeup import get_args, TeethSeg

# global variables
# label_test = '_1'
label_test = ''
today = time.strftime("%y%m%d", time.localtime(time.time()))

if __name__ == '__main__':
    args = get_args()
    if args.ALL:
        args.pseudo_er = True
        args.inits = True
        args.snake = True
        args.id_region = True

    # imgs = args.imgs if args.imgs else [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
    imgs = args.imgs if args.imgs else [0]

    dir_result = join('results', f'er_net/{today}{label_test}/')
    mts.makeDir(dir_result)

    for ni in imgs:
        dir_img = join(dir_result, f'{ni:05d}/')
        mts.makeDir(dir_img)
        sts = mts.SaveTools(dir_img)
        
        # Inference pseudo edge-regions with a deep neural network
        ts = TeethSeg(dir_img, ni, sts, args)
        if args.pseudo_er: ts.pseudoER()
        if args.inits: ts.initContour()
        if args.snake: ts.snake()
        if args.id_region: ts.idReg()
