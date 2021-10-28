from os import stat
from os.path import join

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.arraysetops import unique

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

        if 'tot_lbl' in self.dict.keys():
            self.lbl0 = self.dict['lbl0']
            self.lbl = self.dict['lbl']
            self.er_Fa = self.dict['er_Fa']
            self.lbl_fa = self.dict['lbl_fa']
            self.tot_lbl = self.dict['tot_lbl']
        else:
            self.GADF = GADF(self.img)
            self.Fa = self.GADF.Fa
            self.er_Fa = self.GADF.Er
            self.lbl_er =  label(self.er_Fa, background=0, connectivity=1)
            self.lbl0 = self.labeling()
            self.soaking()
            self.lbl = self.labeling()
            self.lbl_fa = self.toGADF(self.lbl)
            self.tot_lbl = self.zeroReg(self.lbl_fa)
        if 'res' in self.dict.keys():
            self.res = self.dict['res']
        else:
            self.res = self.regClass()
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
            if (lb not in lbl) or (lb == -1):
                continue
            _regs.append(np.where(lbl == (lb), -1., 1.))
            cal_regs.append(np.where((lbl == 0) + (lbl == lb), 1., 0.))

        Rein = Reinitial(dt=.2, width=3.5, tol=0.01)
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

            gx, gy = mts.imgrad(_phis)
            _Fa = - 1 * (gx.transpose((2, 0, 1)) * self.Fa[..., 1] + gy.transpose((2, 0, 1)) * self.Fa[..., 0]) * self.er_Fa * (self.lbl == 0)
            _Fb = np.array([- tg.force() * (1 - self.er_Fa * (self.lbl == 0)) for tg in teg])

            kap = mts.kappa(_phis)[0] * (np.abs(_phis) < 5)
            _F = _Fa + mts.gaussfilt((_Fb).transpose((1, 2, 0)), 1).transpose((2, 0, 1)) * cal_regs + _Fo + mu * kap.transpose((2, 0, 1))
            # _F = (_Fa + _Fb) * cal_regs + _Fo + mu * kap.transpose((2, 0, 1))
            new_phis = _phis + dt * _F.transpose((1, 2, 0))

            err = np.abs(new_phis - _phis).sum() / new_phis.size
            if err < 1E-04 or _k > 200:
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
        self.phi = np.where((self.phi > 0) * (self.er < .5), -1, self.phi)

    def zeroReg(self, lbl):
        '''
        Assign 0 regions by using intensity values
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
        lbl_kapp = self.regClassKapp(self.img, self.tot_lbl)
        cand_rm = self.candVLine(lbl_kapp)
        lbl_vl = self.regClassVLine(self.img, lbl_kapp, cand_rm)
        lbl_sd = self.removeSide(self.img, lbl_vl)
        lbl_sd2 = self.removeBG(lbl_sd)
        return lbl_sd2

    @staticmethod
    def removeBG(lbl):

        if len(np.unique(lbl[[0, -1], :])) == 1 and len(np.unique(lbl[:, [0, -1]])):
            if np.unique(lbl[[0, -1], :])[0] == np.unique(lbl[:, [0, -1]])[0]:
                l = np.unique(lbl[[0, -1], :])[0]
                res = np.copy(lbl)
                res = np.where(lbl == l, -1., res)

        return res

    @staticmethod
    def regClassKapp(img, lbl):
        Rein = Reinitial(dt=.1, width=5)

        reg_nkapp = []
        for l in np.unique(lbl):
            if l < 0: continue
            _reg = np.where(lbl == l, 1., 0.)
            
            # _phi = skfmm.distance(.5 - _reg)
            _phi = Rein.getSDF(.5 - _reg)
            _kapp = mts.kappa(_phi, mode=0)[0]
            _kapp = mts.gaussfilt(_kapp, sig=.5)

            reg_cal = np.abs(_phi) < 1.5
            kapp_p = np.where(_kapp > 1E-04, 1, 0)
            kapp_n = np.where(_kapp < -1E-04, 1, 0)

            n_kapp_p = (kapp_p * reg_cal).sum()
            n_kapp_n = (kapp_n * reg_cal).sum()

            if n_kapp_p < n_kapp_n:
                reg_nkapp.append(l)

        mu_img = np.mean(img, where=np.where(img==0, False, True))
        var_img = np.var(img, where=np.where(img==0, False, True))
        res = np.copy(lbl)
        for rnk in reg_nkapp:
            _reg = (lbl == rnk)
            _mu_r = np.mean(img.transpose((2, 0, 1)), where=_reg)
            if _mu_r <= mu_img:
                res = np.where(lbl == rnk, -1., res)
        return res

    @staticmethod
    def candVLine(lbl):
        reg2bchck = []
        thres = .95
        for l in np.unique(lbl):
            if l < 0: continue
            _reg = np.where(lbl == l)

            # _x = np.unique(_reg[1])
            # n_samples = max(round(len(_x) / 2.), 1)
            _x = _reg[1]
            n_samples = max(round(len(_x) / 20.), 1)
            
            np.random.seed(210501)
            samples_x = np.random.choice(_x, n_samples, replace=False)
            flg = 0
            for s_x in samples_x:
                vl_reg = np.setdiff1d(lbl[:, s_x], [-1, ])
                if len(vl_reg) > 2:
                    flg += 1
            if flg / n_samples > thres:
                reg2bchck.append(l)
        return reg2bchck

    @staticmethod
    def regClassVLine(img, lbl, cand):
        from skimage import color
        from scipy import stats

        res = np.copy(lbl)

        lab = color.rgb2lab(img)
        # max_a = np.unravel_index(np.argmax(lab[..., 1]), lab[..., 1].shape)
        # max_b = np.unravel_index(np.argmax(lab[..., 2]), lab[..., 2].shape)
        # init_k = np.array([[100, 0, 0], lab[max_a[0], max_a[1], :], (lab[max_a[0], max_a[1], :] + [100, 0, 0])/2])
        # init_k = np.array([lab[max_a[0], max_a[1], :], [100, 0, 0]])
        m_a = np.unravel_index(np.argmax(lab[..., 1]), lab[..., 1].shape)
        m_b = np.unravel_index(np.argmax(lab[..., 2]), lab[..., 2].shape)
        # init_k = np.array([[100, 0, 0], p_lab[m_a[0], :], (p_lab[m_a[0], :] + [100, 0, 0])/ 2])
        init_k = np.array([[100, 0, 0], lab[m_a[0], m_a[1], :], [50, 0, lab[m_b[0], m_b[1], 2]]])

        for l in cand:
            if l < 0: continue
            _reg = np.where(lbl == l)

            # _x = np.unique(_reg[1])
            # n_samples = max(round(len(_x) / 2.), 1)
            _x = _reg[1]
            n_samples = max(round(len(_x) / 20.), 1)
            
            np.random.seed(210501)
            samples_x = np.random.choice(_x, n_samples, replace=False)

            modes_reg = []
            for sx in samples_x:
                vl_lab = lab[:, sx]
                vl_img = img[:, sx]
                vl_lbl = lbl[:, sx]

                p_lbl = vl_lbl[vl_lbl >= 0]
                p_img = vl_img[vl_lbl >= 0]
                p_lab = vl_lab[vl_lbl >= 0]

                # m_a = np.unravel_index(np.argmax(p_lab[..., 1]), p_lab[..., 1].shape)
                # m_b = np.unravel_index(np.argmax(p_lab[..., 2]), p_lab[..., 2].shape)
                # init_k = np.array([[100, 0, 0], p_lab[m_a[0], :], [50, 0, p_lab[m_b[0], :][2]]])
                kmeans = KMeans(n_clusters=3, init=init_k).fit(p_lab)
                kmlbl = kmeans.labels_

                l_kmlbl = kmlbl[p_lbl == l]
                # modes_reg.append(int(stats.mode(l_kmlbl)[0]))
                # modes_reg.append(l_kmlbl)
                modes_reg += list(l_kmlbl)

                if 1 == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d') # Axe3D object
                    ax.scatter(p_lab[..., 1], p_lab[..., 2], p_lab[..., 0], vmin=-1, vmax=46, c=p_lbl, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
                    ax.scatter(init_k[..., 1], init_k[..., 2], init_k[..., 0], marker='*', vmax=2, c=[0, 1, 2], s= 60, alpha=1, cmap=mts.colorMapAlpha(plt))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d') # Axe3D object
                    ax.scatter(p_lab[..., 1], p_lab[..., 2], p_lab[..., 0], vmax=2, c=kmlbl, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
                    ax.scatter(init_k[..., 1], init_k[..., 2], init_k[..., 0], marker='*', vmax=2, c=[0, 1, 2], s= 60, alpha=1, cmap=mts.colorMapAlpha(plt))
            
            if int(stats.mode(modes_reg)[0]) == 1:
                res = np.where(res == l, -1, res)
        return res

    @staticmethod
    def removeSide(img, lbl):
        idx = np.where(lbl > 0)
        _r = np.argmax(idx[1])
        _l = np.argmin(idx[1])

        R = lbl[idx[0][_r], idx[1][_r]]
        L = lbl[idx[0][_l], idx[1][_l]]

        res = np.copy(lbl)
        mu = np.mean(img)
        sig = np.sqrt(np.var(img))
        for i in [R, L]:
            _reg = (lbl == i)
            _mu = np.mean(img.transpose((2, 0, 1)), where=_reg)
            if _mu < mu - 1 * sig:
                res = np.where(_reg, -1, res)
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
            Rein = Reinitial(dt=.1)
            ReinKapp = ReinitialKapp(iter=10, mu=.05)
            clrs = ['r'] * 100
            for i in range(int(np.max(contour))):
                _reg = np.where(contour == i+1, -1., 1.)
                for _i in range(10):
                    _reg = Rein.getSDF(_reg)
                    _reg = ReinKapp.getSDF(_reg)
                plt.contour(_reg, levels=[0], colors=clrs[i], linewidths=1)

        plt.axis('off')
        plt.savefig(f'{self.dir_img}{name}', dpi=1024, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _saveSteps(self):
        self._showSaveMax(self.lbl0, 'lbl0.pdf')
        self._showSaveMax(self.lbl, 'lbl.pdf')
        self._showSaveMax(self.er_Fa, 'er_fa.pdf')
        self._showSaveMax(self.lbl_fa, 'lbl_fa.pdf')
        self._showSaveMax(self.tot_lbl, 'tot_lbl.pdf')
        self._showSaveMax(self.res, 'lbl2.pdf')
        self._showSaveMax(self.res, 'lbl2.pdf')
        self._showSaveMax(self.img, 'res_0.pdf', contour=self.res)
        # self._showSaveMax(self.img, 'res_1.pdf', face=self.res)
        # self._showSaveMax(self.img, 'res_2.pdf', face=self.res, contour=self.res)

        self.dict['lbl0'] = self.lbl0
        self.dict['lbl'] = self.lbl
        self.dict['er_Fa'] = self.er_Fa
        self.dict['lbl_fa'] = self.lbl_fa
        self.dict['tot_lbl'] = self.tot_lbl
        self.dict['res'] = self.res


class ReinitialKapp(Reinitial):
    def __init__(self, dt:float=0.1, width:float=5, tol:float=1E-02, iter:int=None, dim:int=2, debug=False, fmm=False, mu=0.1) -> np.ndarray:
        super().__init__(dt=dt, width=width, tol=tol, iter=iter, dim=dim, debug=debug, fmm=fmm)
        self.mu = mu

    def update(self, phi):
        bd, fd = self.imgrad(phi, self.dim)

        # abs(a) and a+ in the paper
        bxa, bxp = np.abs(bd[0]), np.maximum(bd[0], 0)
        # abs(b) and b+ in the paper
        fxa, fxp = np.abs(fd[0]), np.maximum(fd[0], 0)
        # abs(c) and c+ in the paper
        bya, byp = np.abs(bd[1]), np.maximum(bd[1], 0)
        # abs(d) and d+ in the paper
        fya, fyp = np.abs(fd[1]), np.maximum(fd[1], 0)
        if self.dim == 3:
            bza, bzp = np.abs(bd[2]), np.maximum(bd[2], 0)
            fza, fzp = np.abs(fd[2]), np.maximum(fd[2], 0)

        b_sgn, f_sgn = (self.sign0 - 1) / 2, (self.sign0 + 1) / 2

        Gx = np.maximum((bxa * b_sgn + bxp) ** 2, (-fxa * f_sgn + fxp) ** 2)
        Gy = np.maximum((bya * b_sgn + byp) ** 2, (-fya * f_sgn + fyp) ** 2)
        if self.dim == 2:
            _G = np.sqrt(Gx + Gy) - 1
        elif self.dim == 3:
            Gz = np.maximum((bza * b_sgn + bzp) ** 2, (-fza * f_sgn + fzp) ** 2)
            _G = np.sqrt(Gx + Gy + Gz) - 1
        
        # for numerical stabilities, sign should be updated
        _sign0 = self.approx_sign(phi)
        kapp = mts.gaussfilt(mts.kappa(phi)[0], sig=.5)
        # kapp = mts.kappa(phi)[0]
        _phi = phi - self.dt * (_sign0 * _G - self.mu * kapp)
        return _phi