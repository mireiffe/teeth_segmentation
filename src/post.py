from os import stat
from os.path import join

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label
import skfmm
from sklearn.cluster import KMeans

from gadf import GADF
from reinitial import Reinitial
from reinst import ThreeRegions

import myTools as mts


class PostProc():
    eps = np.finfo(float).eps
    jet_alpha = mts.colorMapAlpha(plt)
    
    def __init__(self, dict, dir_img) -> None:
        self.dir_img = dir_img
        self.dict = dict
        self.img = dict['img']
        self.seg_er = dict['seg_er']
        self.er = dict['er']
        self.phi = dict['phi'][..., 0]
        self.m, self.n = self.er.shape

        self.GADF = GADF(self.img)
        self.Fa = self.GADF.Fa
        self.er_Fa = self.GADF.Er

        self.lbl_er =  label(self.er_Fa, background=0, connectivity=1)

        _lbl_er = self.lbl_er * self.er
        temp = np.zeros_like(self.lbl_er)
        ctr = 0
        for i in range(int(self.lbl_er.max())):
            i_r = i + 1
            ids_r = np.where(self.lbl_er == i_r)
            sz_r = len(ids_r[0])
            sz_rer = len(np.where(_lbl_er == i_r)[0])

            if sz_rer / sz_r > ctr:
                temp[ids_r] = 1

        self.lbl0 = self.labeling()
        self.soaking()
        self.lbl = self.labeling()
        self.lbl_fa = self.toGADF(self.lbl)
        self.tot_lbl = self.zeroReg(self.lbl_fa)

        self.res = self.regClass()
        self._saveSteps()

    def toGADF(self, lbl):
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        dt = 0.1
        # mu = .1
        mu = .5
        num_lbl = np.max(lbl) + 1
        X, Y = np.mgrid[0:self.m, 0:self.n]
        _regs = []
        cal_regs = []
        for lb in range(1, num_lbl):
            if (lb not in lbl) or (lb == 0):
                continue
            _regs.append(np.where(lbl == (lb), -1., 1.))
            cal_regs.append(np.where((lbl == 0) + (lbl == lb), 1., 0.))

        Rein = Reinitial(dt=.1, width=5)
        # Rein = Reinitial(dt=.1, width=5, fmm=True)
        _phis = Rein.getSDF(np.transpose(_regs, (1, 2, 0)))
        n_phis = _phis.shape[-1]
        teg = [ThreeRegions(self.img) for nph in range(n_phis)]

        _k = 0
        while True:
            _k += 1
            if _k % 2 == 0:
                _phis = Rein.getSDF(_phis)
                pass
            else:
                pass

            _dist = 1
            regs = np.where(_phis < _dist, _phis - _dist, 0)
            all_regs = regs.sum(axis=-1)
            _Fo = - (all_regs - regs.transpose((2, 0, 1)))

            for _i in range(n_phis):
                teg[_i].setting(_phis[..., _i])

            gx, gy = self.imgrad(_phis)
            _Fa = - 1 * (gx.transpose((2, 0, 1)) * self.Fa[..., 0] + gy.transpose((2, 0, 1)) * self.Fa[..., 1]) * self.er_Fa * (self.lbl == 0)
            _Fb = np.array([- tg.force() * (1 - self.er_Fa) for tg in teg])

            kap = self.kappa(_phis)[0] * (np.abs(_phis) < 5)
            _F = mts.gaussfilt((_Fa + _Fb).transpose((1, 2, 0)), 1).transpose((2, 0, 1)) * cal_regs + _Fo + mu * kap.transpose((2, 0, 1))
            new_phis = _phis + dt * _F.transpose((1, 2, 0))

            err = np.abs(new_phis - _phis).sum() / new_phis.size
            if err < 1E-04 or _k > 20:
                break
        
            if _k % 1 == 0:
                plt.figure(1)
                plt.cla()
                plt.imshow(self.img)
                plt.imshow(self.lbl == 0, 'gray', alpha=.3)
                plt.imshow(self.er_Fa * (self.lbl == 0), alpha=.3)
                for i in range(_phis.shape[-1]):
                    plt.contour(_phis[..., i], levels=[0], colors=clrs[i])
                plt.title(f'iter = {_k:d}')
                # plt.show()
                plt.pause(.1)

            _phis = new_phis

        lbls = np.arange(1, new_phis.shape[-1] + 1)
        return np.dot(np.where(new_phis < 0, 1., 0.), lbls)

    def labeling(self, del_tol=None):
        '''
        labeling connected region (0 value for not assigned region)
        '''
        seg_res = np.where(self.phi < 0, 1., 0.)
        lbl = label(seg_res, background=0, connectivity=1)
        if del_tol is None:
            del_tol = self.m * self.n / 2000
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                lbl[idx_i] = 0

        return lbl

    def soaking(self):
        self.phi = np.where((self.phi > 0) * (self.seg_er < .5), -1, self.phi)

    def zeroReg(self, lbl):
        '''
        Assing 0 regions by using intensity values
        '''
        idx_zero_reg = np.where(lbl == 0)
        _lbl = np.zeros_like(lbl)
        for idx in zip(*idx_zero_reg):
            val_img = self.img[idx]
            _k = 0
            while True:
                _k += 1
                x0 = np.maximum(idx[0]-_k, 0)
                x1 = np.minimum(idx[0]+_k+1, self.m)
                y0 = np.maximum(idx[1]-_k, 0)
                y1 = np.minimum(idx[1]+_k+1, self.n)
                ptch_img = self.img[x0:x1, y0:y1, :]
                ptch = lbl[x0:x1, y0:y1]
                ele_ptch = np.unique(ptch)
                if _k <= 3:
                    if len(ele_ptch) > 2:
                        break
                else:
                    if len(ele_ptch) > 1:
                        break
            min_dist = []
            for ep in ele_ptch[1:]:
                idx_ep = np.where(ptch == ep)
                l2dist = np.sqrt(((val_img - ptch_img[idx_ep])**2).sum(axis=-1))
                min_dist.append(l2dist.min())
            el_min = np.argmin(min_dist) + 1
            _lbl[idx] = ele_ptch[el_min]
                    
        return lbl + _lbl

    def regClass(self):
        lbl_kapp = self.regClassByKapp(self.tot_lbl)
        lbl_res = self.regClassByVLine(lbl_kapp)
        return lbl_res

    @staticmethod
    def regClassByKapp(lbl):
        # Rein = Reinitial(dt=.1, width=4)

        reg_nkapp = []
        for l in np.unique(lbl):
            _reg = np.where(lbl == l, 1., 0.)
            
            # _phi = Rein.getSDF(_reg)
            _phi = skfmm.distance(.5 - _reg)
            _kapp = mts.kappa(_phi, mode=0)[0]
            _kapp = mts.gaussfilt(_kapp, sig=1)

            reg_cal = np.abs(_phi) < 1.5
            kapp_p = np.where(_kapp > 1E-04, _kapp, 0)
            kapp_n = np.where(_kapp < -1E-04, _kapp, 0)

            n_kapp_p = (kapp_p * reg_cal).sum()
            n_kapp_n = (kapp_n * reg_cal).sum()

            if n_kapp_p < n_kapp_n:
                reg_nkapp.append(l)

        res = np.copy(lbl)
        for rnk in reg_nkapp:
            res = np.where(lbl == rnk, -1., res)
        return res

    @staticmethod
    def regClassByVLine(lbl):
        reg2bchck = []
        thres = .95
        for l in np.unique(lbl):
            l = 0
            _reg = np.where(lbl == l)
            n_samples = round(len(_reg[0]) / 10.)
            
            np.random.seed(210501)
            pts_x = np.random.choice(_reg[1], n_samples, replace=False)
            flg = 0
            for pt_x in pts_x:
                vl_reg = lbl[:, pt_x]
                if len(vl_reg) > 2:
                    flg += 1
            if flg / n_samples > thres:
                reg2bchck.append(l)

        res = np.copy(lbl)
        for r2c in reg2bchck:
            res = np.where(lbl == r2c, 0., res)
        return res

    def _showSaveMax(self, obj, name, face=None, contour=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        ax.imshow(obj)
        if face is not None:
            _res = np.where(face < 0, 0, face)
            plt.imshow(_res, alpha=.4, cmap='rainbow_alpha')
        if contour is not None:
            Rein = Reinitial()
            clrs = ['r'] * 100
            for i in range(int(np.max(contour))):
                _reg = np.where(contour == i+1, -1., 1.)
                _reg = Rein.getSDF(_reg)
                plt.contour(_reg, levels=[0], colors=clrs[i], linewidths=1)

        plt.savefig(f'{self.dir_img}{name}', dpi=1024, bbox_inches='tight', facecolor='#eeeeee')
        plt.close(fig)

    def _saveSteps(self):
        self._showSaveMax(self.lbl0, 'lbl0.png')
        self._showSaveMax(self.lbl, 'lbl.png')
        self._showSaveMax(self.er_Fa, 'er_fa.png')
        self._showSaveMax(self.lbl_fa, 'lbl_fa.png')
        self._showSaveMax(self.tot_lbl, 'tot_lbl.png')
        self._showSaveMax(self.res, 'lbl2.png')
        self._showSaveMax(self.res, 'lbl2.png')
        self._showSaveMax(self.img, 'res_0.png', contour=self.res)
        # self._showSaveMax(self.img, 'res_1.png', face=self.res)
        # self._showSaveMax(self.img, 'res_2.png', face=self.res, contour=self.res)

        self.dict['lbl0'] = self.lbl0
        self.dict['lbl'] = self.lbl
        self.dict['er_Fa'] = self.er_Fa
        self.dict['tot_lbl'] = self.tot_lbl
        self.dict['res'] = self.res

        ## meeting

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # Axe3D object
        _img = self.img[::4, ::4, :]
        r,g,b = _img[..., 0], _img[...,1], _img[..., 2]
        # _img = np.stack((0*r, 3 * g, 5 * b), axis=2)
        _tl = self.tot_lbl[::4, ::4]

        _img = np.where(_tl == 1, 0, _img.transpose((2, 0, 1))).transpose((1, 2, 0))
        _img = np.where(np.sum([_tl == v for v in [10, 18, 21, 22, 23, 9, 11]], axis=0), _img.transpose((2, 0, 1)), 0).transpose((1, 2, 0))

        # _img = np.where(_tl == 1, 0, _img.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _img = np.where(_tl == 16, 0, _img.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _img = np.where(np.sum([_tl == v for v in [9, 10, 8]], axis=0), _img.transpose((2, 0, 1)), 0).transpose((1, 2, 0))

        _img = np.where(_tl == 7, 0, _img.transpose((2, 0, 1))).transpose((1, 2, 0))
        _img = np.where(_tl == 1, 0, _img.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _img = np.where(np.sum([_tl == v for v in [9, 10, 8]], axis=0), _img.transpose((2, 0, 1)), 0).transpose((1, 2, 0))
        
        ax.scatter(_img[..., 0], _img[..., 1], _img[..., 2], c=_tl-1, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
        # plt.figure(); plt.imshow(self.tot_lbl, cmap='jet')
        plt.figure(); plt.imshow(self.tot_lbl * (1-np.sum([self.tot_lbl == v for v in [7, 1]], axis=0)), cmap='jet')

        from skimage import io, color
        # lab = color.rgb2lab(self.img)
        lab = color.rgb2lab(self.img)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # Axe3D object
        _lab = lab[::4, ::4, :]
        # _lab = np.where(_tl == 1, 0, _lab.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _lab = np.where(np.sum([_tl == v for v in [10, 18, 21, 22, 23, 9, 11]], axis=0), _lab.transpose((2, 0, 1)), 0).transpose((1, 2, 0))
        
        # _lab = np.where(_tl == 16, 0, _lab.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _lab = np.where(_tl == 1, 0, _lab.transpose((2, 0, 1))).transpose((1, 2, 0))
        # _lab = np.where(np.sum([_tl == v for v in [8, 9, 10, 12, 14]], axis=0), _lab.transpose((2, 0, 1)), 0).transpose((1, 2, 0))
        
        _lab = np.where(_tl == 7, 0, _lab.transpose((2, 0, 1))).transpose((1, 2, 0))
        _lab = np.where(_tl == 1, 0, _lab.transpose((2, 0, 1))).transpose((1, 2, 0))
        
        ax.scatter(_lab[..., 1], _lab[..., 2], _lab[..., 0], c=_tl-1, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))

        plt.figure()
        plt.imshow((lab - lab.min()) / (lab - lab.min()).max())

        ## sk color quant test

        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin
        from sklearn.datasets import load_sample_image
        from sklearn.utils import shuffle
        from time import time

        n_colors = 16

        # china = self.img
        china = lab
        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(china.shape)
        assert d == 3
        image_array = np.reshape(china, (w * h, d))

        print("Fitting model on a small sub-sample of the data")
        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        t0 = time()
        labels = kmeans.predict(image_array)
        print("done in %0.3fs." % (time() - t0))


        codebook_random = shuffle(image_array, random_state=0)[:n_colors]
        print("Predicting color indices on the full image (random)")
        t0 = time()
        labels_random = pairwise_distances_argmin(codebook_random,
                                                image_array,
                                                axis=0)
        print("done in %0.3fs." % (time() - t0))


        def recreate_image(codebook, labels, w, h):
            """Recreate the (compressed) image from the code book & labels"""
            d = codebook.shape[1]
            image = np.zeros((w, h, d))
            label_idx = 0
            for i in range(w):
                for j in range(h):
                    image[i][j] = codebook[labels[label_idx]]
                    label_idx += 1
            return image

        # Display all results, alongside original image
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        plt.title('Original image (96,615 colors)')
        plt.imshow(china)

        plt.figure(2)
        plt.clf()
        plt.axis('off')
        plt.title(f'Quantized image ({n_colors} colors, K-Means)')
        plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

        __img = np.copy(self.img)
        for i in range(int(self.tot_lbl.max())):
            _i = i + 1
            _reg = np.where(self.tot_lbl == _i, True, False)
            _mu = np.mean(self.img.transpose((2, 0, 1)), where=_reg)
            __img = (__img.transpose((2, 0, 1)) - _reg * (_mu - .5)).transpose((1, 2, 0))

        plt.figure()
        plt.imshow(__img)

        lab = color.rgb2lab(__img)
        lab = (lab - lab.min()) / (lab - lab.min()).max()

        plt.figure()
        plt.imshow(lab)

        reg2bchck = []
        for i in range(int(self.tot_lbl.max())):
            _i = i + 1
            _reg = np.where(self.tot_lbl == _i)
            np.random.seed(210501)
            pts = np.random.choice(len(_reg[0]), len(_reg[0]) // 10, replace=False)
            flg = 0
            for pt in pts:
                vl_reg = np.setdiff1d(self.tot_lbl[:, _reg[1][pt]], [7, 1, 2, 3, 34, 35])
                if len(vl_reg) > 2:
                    flg += 1
            if flg / len(pts) > .99:
                reg2bchck.append(_i)

        _sr = np.zeros_like(self.tot_lbl)
        for rr in reg2bchck:
            _sr += np.where(self.tot_lbl == rr, 1, 0) * rr
        plt.figure()
        plt.imshow(_sr, cmap='jet', vmax=43)
        
        vv = 624
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # Axe3D object

        # _img = self.img[:, vv, :]
        # _img = np.where(_sr[:, vv] == 0, 0, _img.transpose((1,0))).transpose((1, 0))

        _img = color.rgb2lab(self.img)[:, vv, :]
        # _img = color.rgb2lab(self.img)[:, vv, :]
        _img = np.where(self.tot_lbl[:, vv] == 7, 0, _img.transpose((1,0))).transpose((1, 0))
        _img = np.where(self.tot_lbl[:, vv] == 1, 0, _img.transpose((1,0))).transpose((1, 0))
        ax.scatter(_img[..., 1], _img[..., 2], _img[..., 0], c=self.tot_lbl[:, vv], s=20, alpha=1, cmap='jet', vmax=43)

        # for i in range(int(self.tot_lbl.max())):
        #     _i = i + 1
        #     _reg = np.where(self.tot_lbl == _i)
        #     np.random.seed(210501)
        #     pts = np.random.choice(len(_reg[0]), 10, replace=False)
        #     for pt in pts:
        #         vl_pt = self.img.mean(axis=2)[:, _reg[1][pt]]

        #         plt.figure()
        #         plt.hist(vl_pt)
