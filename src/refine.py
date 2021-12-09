import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage.measure import label

from gadf import GADF

import myTools as mts
from reinitial import Reinitial


class RefinePreER():
    jet_alpha = mts.colorMapAlpha(plt)
    brg_alpha = mts.colorMapAlpha(plt, _cmap='brg')

    def __init__(self, img, pre_er, sts: mts.SaveTools):
        self.img = img

        self.pre_er = pre_er
        self.measureWidth()

        self.bar_er = np.copy(pre_er)
        self.m, self.n = self.bar_er.shape
        self.len_diag = np.sqrt(self.m**2 + self.n**2)

        self.removeHoleNShorts()
        self.bar_er = mts.imDilErod(
            self.bar_er, rad=max(round(self.wid_er / 1.5), 1),
            kernel_type='circular'
            )
        self.bar_er = mts.imDilErod(
            self.bar_er, rad=max(round(self.wid_er / 1.5), 1),
            kernel_type='circular'
            )
        # sts.imshow(self.bar_er, 'er_pre.pdf', cmap='gray')
        sts.imshows([self.img, self.sk], 'skel_pre.pdf', [None, self.brg_alpha], alphas=[None, None])

        self.num_pts = 10
        self.gap = round(self.wid_er)
        self.maxlen_cv = 2 * (self.gap * (self.num_pts - 1) + self.num_pts)
        # self.maxlen_cv = self.len_diag / 50

        self.endPreSet()
        self.dilErFa()
        plt.figure(); plt.imshow(self.bar_er, 'gray'); plt.imshow(self.erfa, mts.colorMapAlpha(plt)); plt.show()
        self.skeletonize()
        sts.imshow(self.bar_er, 'er_dilerfa.pdf', cmap='gray')
        sts.imshows([self.img, self.sk], 'skel_dilerfa.pdf', [None, self.brg_alpha], alphas=[None, None])

        self.endPreSet()
        self.dilCurve(dim_poly=2)
        self.skeletonize()
        sts.imshow(self.bar_er, 'er_quad.pdf', cmap='gray')
        sts.imshows([self.img, self.sk], 'skel_quad.pdf', [None, self.brg_alpha], alphas=[None, None])

        self.dilCurve(dim_poly=1)
        self.skeletonize()
        sts.imshow(self.bar_er, 'er_lin.pdf', cmap='gray')
        sts.imshows([self.img, self.sk], 'skel_lin.pdf', [None, self.brg_alpha], alphas=[None, None])

        ######################################
        # paper
        ######################################
        # plt.figure()
        # plt.imshow(self.bar_er, 'gray')
        # plt.imshow(self.lbl_Send[32:224, 18:195] > .5, mts.colorMapAlpha(plt))
        # plt.imshow(self.bar_er[32:224, 18:195], 'gray')
        # plt.imshow(self.lbl_Send[32:224, 18:195] > .5, mts.colorMapAlpha(plt))
        # for i, idx in enumerate(self.ind_end):
        #     # if i in [11]:
        #         # continue
        #     xx, yy = list(zip(*idx))
            
        #     # plt.plot(np.array(yy[1::10]) - 19, np.array(xx[1::10]) - 33, 'r-')
        #     # plt.plot(yy[0] - 19, xx[0] - 33, 'b', marker='o', markersize=3)
        #     plt.plot(np.array(yy[0:20]), np.array(xx[0:20]), 'r-')
        #     plt.plot(yy[0], xx[0], 'b', marker='o', markersize=3)
        # plt.axis('off')
        # plt.savefig('forpaper/Fig8_curve.pdf', dpi=1024, bbox_inches='tight', pad_inches=0)
        
        # plt.figure(); plt.imshow(self.bar_er, 'gray'); plt.axis('off'); plt.savefig('forpaper/Fig8_img.pdf', bbox_inches='tight', pad_inches=0)
        

    def endPreSet(self):
        self.skeletonize()
        self.findEndPoints()
        self.findCurves(branch=False)

        # 0 if the end point is available, 1 if not.
        self.flag_end = [1 for ie in self.ind_end]

        Sparam = 2
        Lparam = 5

        self.num_Scut = Sparam * round(self.wid_er)
        self.Scut = self.findEndCut(len_cut=self.num_Scut)
        Scut_er = np.where(self.Scut, 0, self.bar_er)
        lbl_Scuter = label(Scut_er, background=0, connectivity=1)
        self.lbl_Send = np.zeros_like(self.bar_er)
        for i, ie in enumerate(self.ind_end):
            creg = np.where(lbl_Scuter == lbl_Scuter[tuple(zip(ie[0]))], 1, 0)
            if creg.sum() < (3 * self.num_Scut) * self.wid_er:
                self.lbl_Send = np.where(creg, i + 1, self.lbl_Send)

        self.num_Lcut = Lparam * round(self.wid_er)
        self.Lcut = self.findEndCut(len_cut=self.num_Lcut)
        Lcut_er = np.where(self.Lcut, 0, self.bar_er)
        lbl_Lcuter = label(Lcut_er, background=0, connectivity=1)
        self.lbl_Lend = np.zeros_like(self.bar_er)
        for i, ie in enumerate(self.ind_end):
            creg = np.where(lbl_Lcuter == lbl_Lcuter[tuple(zip(ie[0]))], 1, 0)
            if creg.sum() < (3 * self.num_Lcut) * self.wid_er:
                self.lbl_Lend = np.where(creg, i + 1, self.lbl_Lend)
        return 0

    def removeHoleNShorts(self):
        self.removeHoles()
        self.removeShorts()

    def removeShorts(self, param_sz=50):
        self.skeletonize()
        tol_len = self.len_diag/ param_sz
        # lbl_sk = label(self.sk, background=0, connectivity=2)
        lbl_er = label(self.bar_er, background=0, connectivity=2)
        lbl_sk = lbl_er * self.sk
        for l in np.unique(lbl_sk):
            if l == 0: continue
            if np.sum(lbl_sk == l) < tol_len:
                # idx = np.where(lbl_sk == l)
                # self.bar_er = np.where(lbl_er == lbl_er[idx[0][0], idx[1][0]], 0., self.bar_er)
                self.bar_er = np.where(lbl_er == l, 0, self.bar_er)

    def removeHoles(self, input=None, param_sz=100):
        if input is None:
            _er = self.bar_er
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
            self.bar_er = _er
        else:
            return _er
        
                
    def dilErFa(self):
        # make GADF and edge region
        _GADF = GADF(self.img)
        self.fa = _GADF.Fa
        self.erfa = _GADF.Er
        lbl_erfa =  label(self.erfa, background=0, connectivity=1)
        
        lbl_erfa_neter = lbl_erfa * self.bar_er
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
            if (not self.flag_end[_l]) or (l not in lbl_erfa_end):
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
                    add_reg = cv2.filter2D(add_reg, -1, mts.cker(_rad)) > .1
                    _touch = add_reg * np.where(self.lbl_Lend, 0, self.bar_er)
                    if _touch.sum() > 0:
                        self.bar_er = np.where(add_reg, 1, self.bar_er)
                        # self.flag_end[_l] = 0
        return 0

        plt.figure(); plt.imshow(self.er, 'gray'); plt.imshow(use_er, self.jet_alpha, vmax=2)

        selColor = [102, 204, 51]     # Green
        unselColor = [255, 204, 0]    # Yellow
        _er = np.stack([use_erfa * sc for sc in selColor], axis=2) / max(selColor)
        er = np.stack([self.er for sc in selColor], axis=2)
        lev = 50 / 30
        _qimg = np.where(_er, er + lev * _er, er)
        _qimg = (_qimg - _qimg.min()) / (_qimg.max() - _qimg.min())
        res = _qimg[50:209, 16:113]
        plt.imsave('img.png',res)
        plt.imshow(res)

    def preSet(self, branch=False):
        self.skeletonize()
        self.findEndPoints()
        self.findCurves(branch=branch)

    def findEndCut(self, len_cut):
        cut_line = np.zeros_like(self.bar_er)
        for ie in self.ind_end:
            nc = len_cut + 1 if len(ie) > len_cut + 1 else len(ie) - 1
            
            # Find tangent at the cut point
            T = np.array(ie[nc]) - np.array(ie[nc - 2])
            nT = np.sqrt(np.sum(T**2))
            T = T / (nT + mts.eps)
            # T = T / (nT)
            
            Np = np.array([T[1], -T[0]])
            Nn = np.array([-T[1], T[0]])
            n0 = np.array(ie[nc - 1])

            kp, kn = 0, 0
            while True:
                _xp = (n0 + Np * kp / 2).astype(int)
                if np.any(_xp < [0, 0]) or np.any(_xp >= [self.m, self.n]):
                    break

                if self.bar_er[_xp[0], _xp[1]] == 1:
                    cut_line[_xp[0], _xp[1]] = 1
                    kp += 1
                else:
                    break
            while True:
                _xn = (n0 + Nn * kn / 2).astype(int)
                if np.any(_xn < [0, 0]) or np.any(_xn >= [self.m, self.n]):
                    break
                if self.bar_er[_xn[0], _xn[1]] == 1:
                    cut_line[_xn[0], _xn[1]] = 1
                    kn += 1
                else:
                    break
        return cut_line

    def measureWidth(self):
        # sk_idx = np.where(skeletonize(self.bar_er - self.edge_er) == 1)
        sk_idx = np.where(skeletonize(self.pre_er) == 1)
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
                _ptch = self.pre_er[y0:_y+_w+2, x0:_x+_w+2]
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
        # self.psi = rein.getSDF(.5 - self.bar_er)
        # gx, gy = self.imgrad(self.psi)
        # ng = np.sqrt(gx ** 2 + gy ** 2)
        
        # self.sk_phi = np.where((ng < .85) * (self.bar_er > .5), 1., 0.)
        # self.sk = skeletonize(self.sk_phi)
        self.sk = skeletonize(self.bar_er)

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
            if len(idx) < max(self.maxlen_cv // 50, 5):
                lst_del.append(idx)
        for ld in lst_del:
            self.ind_end.remove(ld)

    def dilCurve(self, dim_poly):
        rein = Reinitial()
        self.psi = rein.getSDF(.5  - self.bar_er)

        _gap = .5
        pts = np.arange(0, -self.len_diag, -_gap)
        # pts = np.arange(-self.wid_er * 30, self.wid_er * 30, _gap)
        res = np.zeros_like(self.bar_er)
        debug_res = np.zeros_like(self.bar_er)
        banned = np.zeros((len(self.ind_end), 1))
        filled = np.zeros((len(self.ind_end), 1))
        _er = self.bar_er - self.lbl_Send > .5
        
        if dim_poly == 2:
            lim_prolong = self.len_diag // 25
        elif dim_poly == 1:
            lim_prolong = self.len_diag // 25

        num_reg = label(self.bar_er, background=1, connectivity=1).max()
        for i, ids in enumerate(self.ind_end):
            if not self.flag_end[i]:
                continue
            # _end = np.where(self.lbl_Lend == i + 1)
            # edir = np.array(ids[0])
            # for _k in range(1, min(11, len(ids))): edir = edir - np.array(ids[_k]) / 2**_k
            # edir = edir / np.sqrt(np.sum(edir**2))
            # eidx = list(ids[0])
            # estep = 0.3
            # while True:
            #     eidx += edir * estep
            #     _edx = np.round(eidx).astype(int)
            #     if self.bar_er[_edx[0], _edx[1]] == 0:
            #         break
            # _norm = np.sqrt((_end[0] - _edx[0])**2 + (_end[1] - _edx[1])**2)
            # _as = np.argsort(_norm)

            max_len = int(self.len_diag // 33)
            n_pts = len(ids[:max_len])
            _D = np.arange(n_pts)
            if dim_poly == 1:
                D = np.array([_D, np.ones_like(_D)]).T
            elif dim_poly == 2:
                D = np.array([_D * _D, _D, np.ones_like(_D)]).T

            cv = np.array(list(zip(*ids[:max_len])))

            _res = np.zeros_like(self.bar_er)
            # b = np.take_along_axis(np.array(_end).T, np.stack((_as, _as), axis=1), axis=0)
            b = cv.T
            abc = np.linalg.lstsq(D, b, rcond=None)[0]
            _l = 0

            k0 = 0
            for k, pt in enumerate(pts[1:]):
                if filled[i] or banned[i] or (_l > lim_prolong):
                    continue
                ryy, rxx, _ryy, _rxx = 0, 0, 0, 0
                for kk in range(dim_poly + 1):
                    ryy += abc[-kk-1, 0] * pow(pt, kk)
                    rxx += abc[-kk-1, 1] * pow(pt, kk)
                    _ryy += abc[-kk-1, 0] * pow(pts[k0], kk)
                    _rxx += abc[-kk-1, 1] * pow(pts[k0], kk)
                yy = np.round(ryy).astype(int)
                xx = np.round(rxx).astype(int)
                _yy = np.round(_ryy).astype(int)
                _xx = np.round(_rxx).astype(int)
        
                if (yy == _yy) and (xx == _xx):
                    continue
                if (yy < 0 or xx < 0) or (yy >= self.m or xx >= self.n):
                    banned[i] = 1
                    continue
                if (self.bar_er[_yy, _xx] - self.bar_er[yy, xx]) == -1:
                    filled[i] = 1
                if _res[yy, xx] == 1:
                    continue
                _res[yy, xx] = 1
                _l += 1
                k0 = k

                debug_res = np.where(_res, 1., debug_res)

                _rad = max(round(self.wid_er / 2), 1)
                add_reg = cv2.filter2D(_res, -1, mts.cker(_rad)) > 1E-03
                _touch = add_reg * np.where(self.lbl_Lend == i, 0, self.bar_er)

                if (filled[i]) and (_touch.sum() > 0):
                    _lbl_er = label(self.bar_er, background=1, connectivity=1)
                    _add_lbl = np.unique(add_reg * _lbl_er)
                    _pre_er = np.where(add_reg, 1., self.bar_er)
                    _lbl_per = label(_pre_er, background=1, connectivity=1)

                    sz_lst = 0
                    for al in _add_lbl:
                        if al == 0: continue

                        al_reg = np.where(_lbl_er == al, 1., 0.)
                        al_lbl = label(al_reg * _lbl_per, background=0, connectivity=1)
                        for alb in np.unique(al_lbl):
                            if alb == 0: continue
                            alb_reg = np.where(al_lbl == alb)
                            sz_alb = len(alb_reg[0])

                            tol_back = self.m * self.n / 2000
                            if sz_alb < 3:
                                continue
                            elif sz_alb < tol_back:
                                sz_lst += 1
                                continue
                            elif sz_alb < tol_back * 10:
                                r_x, l_x = np.max(alb_reg[1]), np.min(alb_reg[1])
                                r_y, l_y = np.max(alb_reg[0]), np.min(alb_reg[0])
                                min_wid = np.minimum(np.abs(r_x - l_x) + 1, np.abs(r_y - l_y) + 1)
                                if min_wid < self.wid_er:
                                    sz_lst += 1
                                    continue
                    # if len(np.unique(al_lbl)) - sz_lst - 1 > 1: 
                    if sz_lst == 0: 
                        res += _res
                        self.bar_er = np.where(add_reg, 1., self.bar_er)
                        banned[i] = 1
                        self.flag_end[i] = 0
        return 0

if __name__ == '__main__':
    pass
