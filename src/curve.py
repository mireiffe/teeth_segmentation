from skimage.morphology import skeletonize
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.measure import label

from gadf import GADF
from reinitial import Reinitial
import myTools as mts


class CurveProlong():
    jet_alpha = mts.colorMapAlpha(plt)

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

        self.num_pts = 10
        self.gap = round(self.wid_er)
        self.maxlen_cv = 2 * (self.gap * (self.num_pts - 1) + self.num_pts)

        self.endPreSet()
        self.dilErFa()
        self.skeletonize()
        self.sts.imshow(self.er, 'er_dilerfa.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel_dilerfa.png', [None, self.jet_alpha], alphas=[None, None])

        self.dilCurve(dim_poly=2)
        self.skeletonize()
        self.sts.imshow(self.er, 'er_quad.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel_quad.png', [None, self.jet_alpha], alphas=[None, None])

        self.dilCurve(dim_poly=1)
        self.skeletonize()
        self.sts.imshow(self.er, 'er_lin.png', cmap='gray')
        self.sts.imshows([self.img, self.sk], 'skel_lin.png', [None, self.jet_alpha], alphas=[None, None])

    def endPreSet(self):
        self.skeletonize()
        self.findEndPoints()
        self.findCurves(branch=True)

        # 0 if the end point is available, 1 if not.
        self.flag_end = [1 for ie in self.ind_end]

        Sparam = 2
        Lparam = 5

        self.num_Scut = Sparam * round(self.wid_er)
        self.Scut = self.findEndCut(len_cut=self.num_Scut)
        Scut_er = np.where(self.Scut, 0, self.er)
        lbl_Scuter = label(Scut_er, background=0, connectivity=1)
        self.lbl_Send = np.zeros_like(self.er)
        for i, ie in enumerate(self.ind_end):
            creg = np.where(lbl_Scuter == lbl_Scuter[list(zip(ie[0]))], 1, 0)
            if creg.sum() < (3 * self.num_Scut) * self.wid_er:
                self.lbl_Send += creg * i

        self.num_Lcut = Lparam * round(self.wid_er)
        self.Lcut = self.findEndCut(len_cut=self.num_Lcut)
        Lcut_er = np.where(self.Lcut, 0, self.er)
        lbl_Lcuter = label(Lcut_er, background=0, connectivity=1)
        self.lbl_Lend = np.zeros_like(self.er)
        for i, ie in enumerate(self.ind_end):
            creg = np.where(lbl_Lcuter == lbl_Lcuter[list(zip(ie[0]))], 1, 0)
            if creg.sum() < (3 * self.num_Lcut) * self.wid_er:
                self.lbl_Lend += creg * i
        return 0

    def removeHoleNShorts(self):
        self.measureWidth()
        self.removeHoles()
        self.removeShorts()

    def removeShorts(self, param_sz=10):
        self.skeletonize()
        tol_len = np.sqrt(self.m**2 + self.n**2) / param_sz
        lbl_sk = label(self.sk, background=0, connectivity=2)
        lbl_er = label(self.er, background=0, connectivity=2)
        for _ls in range(int(lbl_sk.max())):
            ls = _ls + 1
            if np.sum(lbl_sk == ls) < tol_len:
                idx = np.where(lbl_sk == ls)
                self.er = np.where(lbl_er == lbl_er[idx[0][0], idx[1][0]], 0., self.er)

    def removeHoles(self, input=None, param_sz=100):
        if input is None:
            _er = self.er
        else:
            _er = input
        lbl_bg = label(_er, background=1, connectivity=1)
        tol_del = self.m * self.n / param_sz**2
        for _l in range(np.max(lbl_bg)):
            l = _l + 1
            ids_l = np.where(lbl_bg == l)
            sz_l = len(ids_l[0])
            if sz_l < tol_del:
                _er[ids_l] = 1
            elif sz_l < tol_del * 10:
                r_x, l_x = np.max(ids_l[1]), np.min(ids_l[1])
                r_y, l_y = np.max(ids_l[0]), np.min(ids_l[0])
                min_wid = np.minimum(np.abs(r_x - l_x) + 1, np.abs(r_y - l_y) + 1)
                if min_wid < self.wid_er:
                    _er[ids_l] = 1
        if input is None:
            self.er = _er
        else:
            return _er
        
                
    def dilErFa(self):
        # make GADF and edge region
        _GADF = GADF(self.img)
        self.erfa = _GADF.Er
        lbl_erfa =  label(self.erfa, background=0, connectivity=1)
        
        lbl_erfa_neter = lbl_erfa * self.er
        use_erfa = np.zeros_like(lbl_erfa)
        ctr = 0.2       # 0 for only touched, 1 for included
        for i in range(int(lbl_erfa.max())):
            i_r = i + 1
            ids_r = np.where(lbl_erfa == i_r)
            sz_r = len(ids_r[0])
            sz_rer = len(np.where(lbl_erfa_neter == i_r)[0])
            if sz_rer / sz_r > ctr:
                use_erfa[ids_r] = 1

        lbl_cuterfa = label(use_erfa * (1 - self.Scut), background=0, connectivity=1)
        lbl_erfa_Send = use_erfa * self.lbl_Send
        lbl_erfa_end = np.zeros_like(lbl_cuterfa)
        for _l in range(int(lbl_erfa_Send.max())):
            l = _l + 1
            if l in lbl_erfa_Send:
                ids_l = np.where(lbl_erfa_Send == l)
                ind_l = np.unique(lbl_cuterfa[ids_l])
                for il in ind_l:
                    lbl_erfa_end = np.where(lbl_cuterfa == il, l, lbl_erfa_end)

        for _l in range(lbl_erfa_end.max()):
            l = _l + 1
            if (not self.flag_end[l]) or (l not in lbl_erfa_end):
                continue
            _regs_end = label(lbl_erfa_end == l, background=0, connectivity=1)
            _regs_Send = _regs_end * (self.lbl_Send > .5)
            for _idx_reg in range(_regs_Send.max()):
                idx_reg = _idx_reg + 1
                _reg = np.where(_regs_Send == idx_reg)
                sz_reg = len(_reg[0])
                if sz_reg >= self.num_Scut / 2:
                    add_reg = np.where(_regs_end == idx_reg, 1., 0.)
                    _rad = max(round(self.wid_er / 2 - 1), 1)
                    add_reg = cv2.filter2D(add_reg, -1, mts.cker(_rad))
                    _touch = add_reg * np.where(self.lbl_Lend, 0, self.er)
                    if _touch.sum() > 0:
                        self.er = np.where(add_reg, 1, self.er)
                        self.flag_end[l] = 0
        return 0

    def preSet(self, branch=False):
        self.skeletonize()
        self.findEndPoints()
        self.findCurves(branch=branch)

    def findEndCut(self, len_cut):
        cut_line = np.zeros_like(self.er)
        for ie in self.ind_end:
            nc = len_cut + 1 if len(ie) > len_cut + 1 else len(ie) - 1
            
            # Find tangent at the cut point
            T = np.array(ie[nc]) - np.array(ie[nc - 2])
            nT = np.sqrt(np.sum(T**2))
            T = T / nT
            
            Np = np.array([T[1], -T[0]])
            Nn = np.array([-T[1], T[0]])
            n0 = np.array(ie[nc - 1])

            kp, kn = 0, 0
            while True:
                _xp = (n0 + Np * kp / 2).astype(int)
                if np.any(_xp < [0, 0]) or np.any(_xp >= [self.m, self.n]):
                    break

                if self.er[_xp[0], _xp[1]] == 1:
                    cut_line[_xp[0], _xp[1]] = 1
                    kp += 1
                else:
                    break
            while True:
                _xn = (n0 + Nn * kn / 2).astype(int)
                if np.any(_xn < [0, 0]) or np.any(_xn >= [self.m, self.n]):
                    break
                if self.er[_xn[0], _xn[1]] == 1:
                    cut_line[_xn[0], _xn[1]] = 1
                    kn += 1
                else:
                    break
        return cut_line

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

    def findEndPoints(self):
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

    def dilCurve(self, dim_poly):
        rein = Reinitial()
        self.psi = rein.getSDF(.5  - self.er)

        _gap = .5
        pts = np.arange(0, -self.wid_er * 20, -_gap)
        # pts = np.arange(-self.wid_er * 30, self.wid_er * 30, _gap)
        res = np.zeros_like(self.er)
        debug_res = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        filled = np.zeros((len(self.ind_end), 1))
        _er = self.er - self.lbl_Lend > .5
        
        lim_prolong = round(self.wid_er * 10)
        num_reg = label(self.er, background=1, connectivity=1).max()
        for i, ids in enumerate(self.ind_end):
            if not self.flag_end[i]:
                continue
            _end = np.where(self.lbl_Lend == i)
            _norm = np.sqrt((_end[0] - ids[0][0])**2 + (_end[1] - ids[0][1])**2)
            _as = np.argsort(_norm)

            n_pts = len(_end[0])
            _D = np.arange(n_pts)
            if dim_poly == 1:
                D = np.array([_D, np.ones_like(_D)]).T
            elif dim_poly == 2:
                D = np.array([_D * _D, _D, np.ones_like(_D)]).T

            _res = np.zeros_like(self.er)
            b = np.take_along_axis(np.array(_end).T, np.stack((_as, _as), axis=1), axis=0)
            abc = np.linalg.lstsq(D, b, rcond=None)[0]
            _l = 0

            for k, pt in enumerate(pts[1:]):
                if filled[i] or banned[i] or (_l > lim_prolong):
                    continue
                yy, xx, _yy, _xx = 0, 0, 0, 0
                for kk in range(dim_poly + 1):
                    yy += abc[-kk-1, 0] * pow(pt, kk)
                    xx += abc[-kk-1, 1] * pow(pt, kk)
                    _yy += abc[-kk-1, 0] * pow(pts[k], kk)
                    _xx += abc[-kk-1, 1] * pow(pts[k], kk)
                yy = np.round(yy).astype(int)
                xx = np.round(xx).astype(int)
                _yy = np.round(_yy).astype(int)
                _xx = np.round(_xx).astype(int)
        
                if (yy == _yy) and (xx == _xx):
                    continue
                if (yy < 0 or xx < 0) or (yy >= self.m or xx >= self.n):
                    banned[i] = 1
                    continue
                if _er[yy, xx] == 1:
                    filled[i] = 1
                _res[yy, xx] = 1
                _l += 1

                debug_res = np.where(_res, 1., debug_res)
                
                _rad = max(round(self.wid_er / 2), 1)
                add_reg = cv2.filter2D(_res, -1, mts.cker(_rad))
                _touch = add_reg * np.where(self.lbl_Lend, 0, self.er)
                if _touch.sum() > 0:
                    res += _res
                    self.er = np.where(add_reg, 1, self.er)
                    banned[i] = 1
                    self.flag_end[i] = 0
        return 0

if __name__ == '__main__':
    pass