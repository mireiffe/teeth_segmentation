from skimage.morphology import skeletonize
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.measure import label

from gadf import GADF
from reinitial import Reinitial
import myTools as mts


class CurveProlong():
    jet_alpha = mts.ColorMapAlpha(plt).name
    num_pts = 10

    def __init__(self, img, er, dir_save):
        self.img = img

        self.er0 = np.copy(er)
        self.er = np.ones_like(er)
        self.er[2:-2, 2:-2] = self.er0[2:-2, 2:-2]
        self.edge_er = self.er - self.er0
        self.m, self.n = self.er.shape

        self.dir_save = dir_save
        self.sts = mts.SaveTools(dir_save)
        
        self.removeHoleNShorts()
        self.sts.imshow(self.er, 'er_pre.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel_pre.png', [None, self.jet_alpha], alphas=[None, None])

        self.gap = round(self.wid_er)
        self.maxlen_cv = 2 * (self.gap * (self.num_pts - 1) + self.num_pts)

        self.dilErFa()
        self.preSet(branch=True)
        self.sts.imshow(self.er, 'er_dilerfa.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel_dilerfa.png', [None, self.jet_alpha], alphas=[None, None])

        self.dilCurve()
        self.preSet()
        self.sts.imshow(self.er, 'er.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel.png', [None, self.jet_alpha], alphas=[None, None])

    def removeHoleNShorts(self):
        self.measureWidth()
        self.removeHoles()
        self.removeShorts()

    def dilErFa(self):
        self.preSet(branch=True)
        self.measureWidth()
        
        # make GADF and edge region
        _GADF = GADF(self.img)
        self.erfa = _GADF.Er
        lbl_erfa =  label(self.erfa, background=0, connectivity=1)
        
        lbl_erfa_neter = lbl_erfa * self.er
        use_erfa = np.zeros_like(lbl_erfa)
        ctr = 0.3
        for i in range(int(lbl_erfa.max())):
            i_r = i + 1
            ids_r = np.where(lbl_erfa == i_r)
            sz_r = len(ids_r[0])
            sz_rer = len(np.where(lbl_erfa_neter == i_r)[0])

            if sz_rer / sz_r > ctr:
                use_erfa[ids_r] = 1

        num_cut = 2 * round(self.wid_er)
        cut_pix = self.findEndCut(num_cut=num_cut)

        lbl_cuter = label(np.where(cut_pix, 0, self.er), background=0, connectivity=1)
        cut_lbl = np.zeros_like(lbl_erfa)
        for ie in self.ind_end:
            creg = np.where(lbl_cuter == lbl_cuter[list(zip(ie[0]))], 1, 0)
            if creg.sum() < 5 * self.wid_er * num_cut:
                cut_lbl += creg * lbl_cuter[list(zip(ie[0]))]

        num_cut2 = 3 * round(self.wid_er)
        cut_pix2 = self.findEndCut(num_cut=num_cut2)

        lbl_cuter2 = label(np.where(cut_pix2, 0, self.er), background=0, connectivity=1)
        cut_lbl2 = np.zeros_like(lbl_erfa)
        for ie in self.ind_end:
            creg = np.where(lbl_cuter2 == lbl_cuter2[list(zip(ie[0]))], 1, 0)
            if creg.sum() < 5 * self.wid_er * num_cut:
                cut_lbl2 += creg * lbl_cuter2[list(zip(ie[0]))]

        er_cut = (cut_lbl > .5) * use_erfa
        er_end_lbl = label(self.erfa * (1 - cut_pix), background=0, connectivity=1)

        for cl in range(cut_lbl.max()):
            _cl = cl + 1
            _reg = (cut_lbl == _cl) * er_cut
            _lbl = (er_end_lbl * _reg).max()
            sz_reg = np.sum(_reg)
            if sz_reg >= num_cut // 2:
                add_reg = np.where(er_end_lbl == _lbl, 1., 0.)
                _rad = round(self.wid_er / 2) + 1
                add_reg = cv2.filter2D(add_reg, -1, self._ker(_rad))
                if (add_reg * (self.er - (cut_lbl2 > .5))).sum() > 0:
                    self.er = np.where(add_reg, 1., self.er)

    def preSet(self, branch=False):
        self.skeletonize()
        self.endPoints()
        self.findCurves(branch=branch)

    def reSet(self, k):
        # self.sk = skeletonize(self.sk)

        # _dil = cv2.filter2D(self.sk.astype(float), -1, self.ker) > 0.1
        _dil = cv2.filter2D(self.new_er.astype(float), -1, self._ker(self.wid_er / 2)) > 0.1
        _sk = skeletonize(np.where(_dil, 1., self.er)).astype(float)
        dil_ker = np.ones((2*round(self.wid_er)+1, 2*round(self.wid_er)+1))
        self.er = np.where(self.er < .5, cv2.filter2D(_sk, -1, self._ker(self.wid_er / 2)) > .1, self.er)
        
        self.skeletonize()
        self.endPoints()
        self.findCurves()
        self.smallGap()
        self.skeletonize()
        self.endPoints()
        self.findCurves(branch=True)

    def findEndCut(self, num_cut):
        cut_pix = np.zeros_like(self.er)
        for ie in self.ind_end:
            nc = num_cut+1 if len(ie) > num_cut + 1 else len(ie) - 1
            _t = np.array(ie[nc]) - np.array(ie[nc - 2])
            _nt = np.sqrt(np.sum(_t**2))
            _t = _t / _nt
            _np = np.array([_t[1], -_t[0]])
            _nn = np.array([-_t[1], _t[0]])
            _n0 = np.array(ie[nc - 1])

            _kp, _kn = 0, 0
            while True:
                _xp = (_n0 + _np * _kp / 2).astype(int)

                if np.any(_xp < [0, 0]) or np.any(_xp >= [self.m, self.n]):
                    break

                if self.er[_xp[0], _xp[1]] == 1:
                    cut_pix[_xp[0], _xp[1]] = 1
                    _kp += 1
                else:
                    break
            while True:
                _xn = (_n0 + _nn * _kn / 2).astype(int)
                if np.any(_xn < [0, 0]) or np.any(_xn >= [self.m, self.n]):
                    break

                if self.er[_xn[0], _xn[1]] == 1:
                    cut_pix[_xn[0], _xn[1]] = 1
                    _kn += 1
                else:
                    break
        return cut_pix

    def smallGap(self):
        num_cut = 10 * round(self.wid_er)
        cuts = self.findEndCut(num_cut=num_cut)
        lbl_cutpix = label(np.where(cuts, 0, self.er), background=0, connectivity=1)
        cut_lbl = np.zeros_like(self.er)
        for ie in self.ind_end:
            _ind = lbl_cutpix == lbl_cutpix[list(zip(ie[0]))]
            if _ind.sum() < 5 * num_cut * self.wid_er:
                cut_lbl = np.where(lbl_cutpix == lbl_cutpix[list(zip(ie[0]))], lbl_cutpix[list(zip(ie[0]))], cut_lbl)
        cut_end = np.where(cut_lbl > .5, 1., 0)
        _dil = cv2.filter2D(cut_lbl, -1, self._ker(self.wid_er / 2)) > 0.1
        _sk = skeletonize(np.where(_dil, 1., self.er)).astype(float)
        dil_ker = np.ones((2*round(self.wid_er)+1, 2*round(self.wid_er)+1))
        _res = np.where(self.er < .5, cv2.dilate(_sk, dil_ker, iterations=1), self.er)
        # self.er = _res

        _gap = .5
        pts = np.arange(100, -self.wid_er * 30, -_gap)
        # pts = np.arange(-self.wid_er * 30, self.wid_er * 30, _gap)
        res = np.zeros_like(self.er)
        ress = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        filled = np.zeros((len(self.ind_end), 1))
        _er = self.er - cut_lbl

        lim_prolong = round(self.wid_er * 2)
        for iii, idx in enumerate(self.ind_end):
            _lbl = cut_lbl[list(zip(idx[0]))]
            _end = np.where(cut_lbl == _lbl)
            _norm = np.sqrt((_end[0] - idx[0][0])**2 + (_end[1] - idx[0][1])**2)
            _as = np.argsort(_norm)
            n_pts = len(_end[0])
            _D = np.arange(n_pts)
            D = np.array([_D * _D, _D, np.ones_like(_D)]).T
            _res = np.zeros_like(self.er)
            b = np.take_along_axis(np.array(_end).T, np.stack((_as, _as), axis=1), axis=0)
            abc = np.linalg.lstsq(D, b, rcond=None)[0]
            # abc[-1, :] = idx[0]
            _l = 0
            for k, pt in enumerate(pts[1:]):
                yy = np.round(abc[0, 0] * pt * pt + abc[1, 0] * pt + abc[2, 0]).astype(int)
                xx = np.round(abc[0, 1] * pt * pt + abc[1, 1] * pt + abc[2, 1]).astype(int)

                _yy = np.round(abc[0, 0] * pts[k] * pts[k] + abc[1, 0] * pts[k] + abc[2, 0]).astype(int)
                _xx = np.round(abc[0, 1] * pts[k] * pts[k] + abc[1, 1] * pts[k] + abc[2, 1]).astype(int)

                if (yy == _yy) and (xx == _xx):
                    continue
                if yy < 0 or xx < 0:
                    banned[iii] = 1
                    continue
                if yy >= self.m or xx >= self.n:
                    banned[iii] = 1
                    continue
                if cut_end[yy, xx]:
                    continue
                if _er[yy, xx] == 1:
                    filled[iii] = 1
                if _l <= lim_prolong:
                    _res[yy, xx] = 1
                    _l += 1
            if filled[iii]:
                res += _res
            ress += _res

        if np.abs(res).sum() > 0:
            dil_res = cv2.filter2D(res, -1, self._ker(self.wid_er / 2)) > .1
            self.er = np.where(self.er < .5, dil_res, self.er)
            
    def measureWidth(self):
        sk_idx = np.where(skeletonize(self.er - self.edge_er) == 1)
        tot_len = len(sk_idx[0])
        np.random.seed(900314)
        sel_idx = np.random.choice(tot_len, tot_len // 10, replace=False)

        wid_er = []
        for si in sel_idx:
            _w = 0
            _x = sk_idx[1][si]
            _y = sk_idx[0][si]
            while True:
                y0 = _y-_w-1 if _y-_w-1 >= 0 else None
                x0 = _x-_w-1 if _x-_w-1 >= 0 else None
                _ptch = self.er0[y0:_y+_w+2, x0:_x+_w+2]
                if _ptch.sum() < _ptch.size:
                    wid_er.append(2*_w + 1)
                    break
                else:
                    _w += 1
        mu = sum(wid_er) / len(sel_idx)
        sig = np.std(wid_er)
        Z_45 = 1.65     # standard normal value for 90 %
        self.wid_er = Z_45 * sig / np.sqrt(tot_len // 10) + mu

    def dilation(self, wid_er, k=0):
        self.dil_ends = np.zeros_like(self.er)
        for idx in self.ind_end:
            self.dil_ends[idx[0]] = 1

        # wid_er = round((self.m + self.n) / 2 / 300)
        rad = int(3 * wid_er)
        Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
        cen_pat = rad
        ker = np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)
        self.dil_ends = cv2.filter2D(self.dil_ends, -1, kernel=ker, borderType=cv2.BORDER_REFLECT)

        # _ends = np.where(_pre == 1, self.er, 0)
        # dil_ends = cv2.dilate(_ends, np.ones((3, 3)), iterations=2)

        plt.figure()
        plt.imshow(self.er, 'gray')
        plt.imshow(self.sk, alpha=.5)
        plt.imshow(self.dil_ends, 'Reds', alpha=.5)
        plt.savefig(f'{self.dir_save}ends{k}.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        self.er = np.where(self.dil_ends + self.er > .5, 1., 0.)
        self.sk = skeletonize(self.er)

    def removeShorts(self, param_del=10):
        self.skeletonize()
        tol_len = np.sqrt(self.m**2 + self.n**2) / param_del
        lbl_sk = label(self.sk, background=0, connectivity=2)
        lbl_er = label(self.er, background=0, connectivity=2)
        for _ls in range(int(lbl_sk.max())):
            ls = _ls + 1
            if np.sum(lbl_sk == ls) < tol_len:
                idx = np.where(lbl_sk == ls)
                self.er = np.where(lbl_er == lbl_er[idx[0][0], idx[1][0]], 0., self.er)

    def removeHoles(self, param_sz=100):
        lbl = label(self.er, background=1, connectivity=1)
        del_tol = self.m * self.n / param_sz**2
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            max_x, min_x = np.max(idx_i[1]), np.min(idx_i[1])
            max_y, min_y = np.max(idx_i[0]), np.min(idx_i[0])
            min_wid = np.minimum(np.abs(max_x - min_x) + 1, np.abs(max_y - min_y) + 1)
            if num_i < del_tol:
                self.er[idx_i] = 1
            elif (num_i < del_tol * 10) and min_wid < self.wid_er:
                self.er[idx_i] = 1

    def skeletonize(self):
        '''
        skeletonization of edge region
        '''
        # rein = Reinitial()
        # self.psi = rein.getSDF(.5 - self.er)
        # gx, gy = self.imgrad(self.psi)
        # ng = np.sqrt(gx ** 2 + gy ** 2)
        
        # self.sk_phi = np.where((ng < .85) * (self.er > .5), 1., 0.)
        # self.sk = skeletonize(self.sk_phi)
        self.sk = skeletonize(self.er)

    def endPoints(self):
        # find end points
        _ind = np.where(self.sk > .5)
        _end = np.zeros_like(self.sk)
        self.ind_end = []
        for iy, ix in zip(*_ind):
            _ptch = self.sk[iy - 1:iy + 2, ix - 1:ix + 2]
            if (_ptch.sum() < 3) and (_ptch.sum() > 0):
                self.ind_end.append([(iy, ix)])
                _end[iy, ix] = 1

    def findCurves(self, maxlen=None, branch=False):
        if maxlen is None:
            maxlen = self.maxlen_cv
        # find curves
        for idx in self.ind_end:
            y0, x0 = idx[0]
            for y_i, x_i in idx:
                ptch = self.sk[y_i-1:y_i+2, x_i-1:x_i+2]
                ind_ptch = np.where(ptch > .5)
                if (not branch) and (len(ind_ptch[0]) > 3): 
                    continue
                for yy_i, xx_i in zip(*ind_ptch):
                    _pre = (yy_i + y_i - 1 != y0) or (xx_i + x_i - 1!= x0)
                    _curr = (yy_i != 1) or (xx_i != 1)
                    if _pre and _curr:
                        idx.append((y_i + yy_i - 1, x_i + xx_i - 1))
                y0, x0 = y_i, x_i
                if len(idx) >= maxlen:
                    break

        lst_del = []
        for idx in self.ind_end:
            if len(idx) < self.maxlen_cv // 3:
                lst_del.append(idx)
        for ld in lst_del:
            self.ind_end.remove(ld)

    def dilCurve(self):
        rein = Reinitial()
        self.psi = rein.getSDF(.5  - self.er)

        reinkapp = ReinKapp(dt=0.1, mu=0.01)

        num_cut = round(5 * self.wid_er)
        cut_pix = self.findEndCut(num_cut)

        cut_er = np.where(cut_pix, 0, self.er)
        lbl_cuter = label(cut_er, background=0, connectivity=1)
        lbl_ender = np.zeros_like(cut_pix)
        for ie in self.ind_end:
            creg = np.where(lbl_cuter == lbl_cuter[list(zip(ie[0]))], 1, 0)
            if creg.sum() < 3 * self.wid_er * num_cut:
                lbl_ender += creg * lbl_cuter[list(zip(ie[0]))]

        self.psi_end = rein.getSDF(np.where(lbl_ender > .5, -1, 1))
        _k = 0
        while True:
            _k += 1
            _kapp = self.gaussfilt(reinkapp.kappa(self.psi_end)[0], sig=.5)
            self.psi_end += 0.01 * _kapp / _kapp.max()
            if _k % 5 == 0:
                self.psi_end = rein.getSDF(self.psi_end)
            if _k > 00:
                break

        cut_lbl = lbl_ender * (self.psi_end < 0)
        # cut_lbl = lbl_ender

        _gap = .5
        pts = np.arange(0, -self.wid_er * 20, -_gap)
        # pts = np.arange(-self.wid_er * 30, self.wid_er * 30, _gap)
        res = np.zeros_like(self.er)
        ress = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        filled = np.zeros((len(self.ind_end), 1))
        _er = self.er - lbl_ender > .5
        
        lim_prolong = round(self.wid_er * 2)
        for iii, idx in enumerate(self.ind_end):
            _lbl = lbl_ender[list(zip(idx[0]))]
            _end = np.where(cut_lbl == _lbl)
            _norm = np.sqrt((_end[0] - idx[0][0])**2 + (_end[1] - idx[0][1])**2)
            _as = np.argsort(_norm)
            n_pts = len(_end[0])
            _D = np.arange(n_pts)
            D = np.array([_D * _D, _D, np.ones_like(_D)]).T
            _res = np.zeros_like(self.er)
            b = np.take_along_axis(np.array(_end).T, np.stack((_as, _as), axis=1), axis=0)
            abc = np.linalg.lstsq(D, b, rcond=None)[0]
            # abc[-1, :] = idx[0]
            _l = 0
            for k, pt in enumerate(pts[1:]):
                if filled[iii] or banned[iii]:
                    continue
                yy = np.round(abc[0, 0] * pt * pt + abc[1, 0] * pt + abc[2, 0]).astype(int)
                xx = np.round(abc[0, 1] * pt * pt + abc[1, 1] * pt + abc[2, 1]).astype(int)

                _yy = np.round(abc[0, 0] * pts[k] * pts[k] + abc[1, 0] * pts[k] + abc[2, 0]).astype(int)
                _xx = np.round(abc[0, 1] * pts[k] * pts[k] + abc[1, 1] * pts[k] + abc[2, 1]).astype(int)

                if (yy == _yy) and (xx == _xx):
                    continue
                if yy < 0 or xx < 0:
                    banned[iii] = 1
                    continue
                if yy >= self.m or xx >= self.n:
                    banned[iii] = 1
                    continue
                #if (cut_lbl > 0.5)[yy, xx]:
                #    continue
                if _er[yy, xx] == 1:
                    filled[iii] = 1
                if _l <= lim_prolong:
                    _res[yy, xx] = 1
                    _l += 1
                _res[yy, xx] = 1
            if filled[iii]:
                res += _res
            ress += _res

        self.er = (self.er + cv2.filter2D(res, -1, self._ker(round(self.wid_er / 2)))) > .5

        xxx = 1
                
    @staticmethod
    def imgrad(img: np.ndarray) -> np.ndarray:
        # ksize = 1: central, ksize = 3: sobel, ksize = -1:scharr
        gx = cv2.Sobel(img, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
        gy = cv2.Sobel(img, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
        return gx / 2, gy / 2

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

    @staticmethod
    def _ker(rad):
        rad = np.maximum(round(rad), 1)
        Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
        cen_pat = rad
        return np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)


class ReinKapp(Reinitial):
    def __init__(self, dt:float=0.1, mu:float=0.01, width:float=30, tol:float=1E-02, iter:int=None, dim:int=2, debug=False, fmm=False) -> np.ndarray:
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
        _kapp = self.gaussfilt(self.kappa(phi)[0], sig=1)
        _phi = phi - self.dt * (_sign0 * _G - self.mu * _kapp)
        return _phi

    def kappa(self, phis, ksz=1, h=1, mode=0):
        x, y = self.imgradKapp(phis)
        if mode == 0:
            ng = np.sqrt(x**2 + y**2 + self.eps)
            nx, ny = x / ng, y / ng
            xx, _ = self.imgradKapp(nx)
            _, yy = self.imgradKapp(ny)
            return xx + yy, x, y, ng
        elif mode == 1:
            xx, yy, xy = self.imgradKapp(phis, order=2)
            res_den = xx * y * y - 2 * x * y * xy + yy * x * x
            res_num = np.power(x ** 2 + y ** 2, 1.5)
            ng = np.sqrt(x**2 + y**2 + self.eps)        # just for output
            return res_den / (res_num + self.eps), x, y, ng

    @staticmethod
    def imgradKapp(img, order=1, h=1) -> np.ndarray:
        '''
        central difference
        '''
        if order == 1:
            _x_ = img[:, 2:, ...] - img[:, :-2, ...]
            x_ = img[:, 1:2, ...] - img[:, :1, ...]
            _x = img[:, -1:, ...] - img[:, -2:-1, ...]

            _y_ = img[2:, :, ...] - img[:-2, :, ...]
            y_ = img[1:2, :, ...] - img[:1, :, ...]
            _y = img[-1:, :, ...] - img[-2:-1, :, ...]

            gx = np.concatenate((x_, _x_, _x), axis=1)
            gy = np.concatenate((y_, _y_, _y), axis=0)
            return gx / (2 * h), gy / (2 * h)
        elif order == 2:
            _img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='symmetric')

            gxx = _img[1:-1, 2:, ...] + _img[1:-1, :-2, ...] - 2 * _img[1:-1, 1:-1, ...]
            gyy = _img[2:, 1:-1, ...] + _img[:-2, 1:-1, ...] - 2 * _img[1:-1, 1:-1, ...]
            gxy = _img[2:, 2:, ...] + _img[:-2, :-2, ...] - _img[2:, :-2, ...] - _img[:-2, 2:, ...]
            return gxx / (h * h), gyy / (h * h), gxy / (4 * h * h)

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)



if __name__ == '__main__':
    pass