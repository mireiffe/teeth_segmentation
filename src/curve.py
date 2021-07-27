from skimage.morphology import skeletonize
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.measure import label

from gadf import GADF
from reinitial import Reinitial


class CurveProlong():
    num_pts = 10

    def __init__(self, er, img, dir_save):
        self.er0 = np.copy(er)
        self.er = np.ones_like(er)
        self.er[2:-2, 2:-2] = self.er0[2:-2, 2:-2]
        self.edge_er = self.er - self.er0

        self.img = img
        self.dir_save = dir_save
        self.m, self.n = self.er.shape
    
        self.gap = np.maximum(round(np.round(self.m * self.n / 300 / 300)), 1)
        self.maxlen_cv = 2 * (self.gap * (self.num_pts - 1) + self.num_pts)
        self.wid_er = self.measureWidth()

        rad = round(2 * self.wid_er)
        Y, X = np.indices([2 * rad + 1, 2 * rad + 1])
        cen_pat = rad
        ker = np.where((X - cen_pat)**2 + (Y - cen_pat)**2 <= rad**2, 1., 0.)
        self.ker = ker

        self.removeHoles()
        self.skeletonize()
        self.removeShorts()

        self.skeletonize()
        self.endPoints()
        self.findCurves()
        
        plt.figure()
        plt.imshow(self.img)
        plt.imshow(self.sk, 'gray', alpha=.5)
        plt.savefig(f'{self.dir_save}skel0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        self.smallReg()
        self.smallGap()

        plt.figure()
        plt.imshow(self.er, 'gray')
        plt.savefig(f'{self.dir_save}er_mid.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        
        self.skeletonize()
        self.endPoints()
        self.findCurves()

        plt.figure()
        plt.imshow(self.img)
        plt.imshow(self.sk, 'gray', alpha=.5)
        plt.savefig(f'{self.dir_save}skel_mid.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        self.GADF = GADF(self.img)
        self.Fa = self.GADF.Fa
        self.er_Fa = self.GADF.Er

        self.lbl_er =  label(self.er_Fa, background=0, connectivity=1)

        _lbl_er = self.lbl_er * self.er
        temp = np.zeros_like(self.lbl_er)
        ctr = 0.5
        for i in range(int(self.lbl_er.max())):
            i_r = i + 1
            ids_r = np.where(self.lbl_er == i_r)
            sz_r = len(ids_r[0])
            sz_rer = len(np.where(_lbl_er == i_r)[0])

            if sz_rer / sz_r > ctr:
                temp[ids_r] = 1

        num_cut = 21
        cut_pix = self.findEndCut(num_cut=num_cut)

        lbl_cutpix = label(np.where(cut_pix, 0, self.er), background=0, connectivity=1)
        cut_end = np.zeros_like(self.lbl_er)
        cut_lbl = np.zeros_like(self.lbl_er)
        for ie in self.ind_end:
            cut_end = np.where(lbl_cutpix == lbl_cutpix[list(zip(ie[0]))], 1., cut_end)
            cut_lbl = np.where(lbl_cutpix == lbl_cutpix[list(zip(ie[0]))], lbl_cutpix[list(zip(ie[0]))], cut_lbl)

        er_cut = cut_end * temp
        er_end_lbl = label(self.er_Fa * (1 - cut_pix), background=0, connectivity=1)
        for cl in range(cut_lbl.max()):
            _cl = cl + 1
            _reg = (cut_lbl == _cl) * er_cut
            _lbl = (er_end_lbl * _reg).max()
            sz_reg = np.sum(_reg)
            if sz_reg >= num_cut // 2:
                add_reg = np.where(er_end_lbl == _lbl, 1., 0.)
                add_reg = cv2.dilate(add_reg, np.ones((3, 3)), iterations=1)
                self.er = np.where(add_reg, 1., self.er)

            
        # temp2 = np.zeros_like(self.lbl_er)
        # temp3 = np.zeros_like(self.lbl_er)
        # for ie in self.ind_end:
        #     temp2[list(zip(*ie[:1]))] = 1
        #     temp3[list(zip(*ie[:20]))] = 1
        # temp2 = cv2.filter2D(temp2.astype(float), -1, ker) > 0.1
        # temp3 = cv2.filter2D(temp3.astype(float), -1, ker) > 0.1

        # plt.figure(); plt.imshow(self.er, 'gray'); plt.imshow(cut_end, 'rainbow_alpha'); plt.imshow(temp, 'rainbow_alpha', vmax=2); plt.show()

        # plt.figure()
        # plt.imshow(self.er, 'gray')
        # plt.imshow(temp, 'rainbow_alpha')
        # plt.imshow(self.sk, 'rainbow_alpha', vmax=5, alpha=3)
        # plt.imshow(self.sk_phi, 'rainbow_alpha', vmax=5, alpha=3)
        # plt.imshow(temp2, 'rainbow_alpha', vmax=2, alpha=1)
        # plt.imshow(temp3, 'rainbow_alpha', vmax=3, alpha=1)
        # plt.show()

        # self.dilation(wid_er=self.wid_er)

        self.smallGap()
        self.skeletonize()
        self.endPoints()
        self.findCurves(branch=True)
        
        plt.figure()
        plt.imshow(self.er, 'gray')
        plt.savefig(f'{self.dir_save}er.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        plt.figure()
        plt.imshow(self.img)
        plt.imshow(self.sk, 'gray', alpha=.5)
        plt.savefig(f'{self.dir_save}skel.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')


    def reSet(self, k):
        # self.sk = skeletonize(self.sk)

        # _dil = cv2.filter2D(self.sk.astype(float), -1, self.ker) > 0.1
        _dil = cv2.filter2D(self.new_er.astype(float), -1, self.ker) > 0.1
        _sk = skeletonize(np.where(_dil, 1., self.er)).astype(float)
        dil_ker = np.ones((2*round(self.wid_er)+1, 2*round(self.wid_er)+1))
        self.er = np.where(self.er < .5, cv2.dilate(_sk, dil_ker, iterations=1), self.er)
        
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
            # _n1 = np.array(ie[num_cut + 1])

            _kp, _kn = 0, 0
            while True:
                _xp = (_n0 + _np * _kp / 2).astype(int)
                # _xp1 = (_n1 + _np * _kp / 2).astype(int)

                if np.any(_xp < [0, 0]) or np.any(_xp >= [self.m, self.n]):
                    break

                if self.er[_xp[0], _xp[1]] == 1:
                    cut_pix[_xp[0], _xp[1]] = 1
                    # cut_pix[_xp1[0], _xp1[1]] = 0
                    _kp += 1
                else:
                    break
            while True:
                _xn = (_n0 + _nn * _kn / 2).astype(int)
                if np.any(_xn < [0, 0]) or np.any(_xn >= [self.m, self.n]):
                    break

                if self.er[_xn[0], _xn[1]] == 1:
                    cut_pix[_xn[0], _xn[1]] = 1
                    # cut_pix[_xn1[0], _xn1[1]] = 0
                    _kn += 1
                else:
                    break
        return cut_pix

    def smallGap(self):
        num_cut = 2 * round(self.wid_er)
        cuts = self.findEndCut(num_cut=num_cut)
        lbl_cutpix = label(np.where(cuts, 0, self.er), background=0, connectivity=1)
        cut_end = np.zeros_like(self.er)
        for ie in self.ind_end:
            _ind = lbl_cutpix == lbl_cutpix[list(zip(ie[0]))]
            if _ind.sum() < 5 * num_cut * self.wid_er:
                cut_end = np.where(lbl_cutpix == lbl_cutpix[list(zip(ie[0]))], 1., cut_end)
        _dil = cv2.filter2D(cut_end, -1, self.ker) > 0.1
        _sk = skeletonize(np.where(_dil, 1., self.er)).astype(float)
        dil_ker = np.ones((2*round(self.wid_er)+1, 2*round(self.wid_er)+1))
        _res = np.where(self.er < .5, cv2.dilate(_sk, dil_ker, iterations=1), self.er)
        # self.er = _res

        n_pts = 15
        gap = int(self.wid_er * 2.5)
        _gap = .5
        pts = np.arange(0, -gap, -_gap)
        res = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        filled = np.zeros((len(self.ind_end), 1))
        _er = self.er - cut_end

        _D = np.arange(n_pts)
        D = np.array([_D * _D, _D, np.ones_like(_D)]).T
        for iii, idx in enumerate(self.ind_end):
            _res = np.zeros_like(self.er)
            for k, pt in enumerate(pts[1:]):
                if (len(idx) < n_pts) or banned[iii] or filled[iii]:
                    continue
                b = np.array(list(zip(*idx[:n_pts]))).T
                abc = np.linalg.lstsq(D, b, rcond=None)[0]
                abc[-1, :] = b[0]

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
                if _er[yy, xx] == 1:
                    filled[iii] = 1
                _res[yy, xx] = 1
            if filled[iii]:
                res += _res

        if np.abs(res).sum() > 0:
            dil_ker = np.ones((2*round(self.wid_er)+1, 2*round(self.wid_er)+1))
            dil_res = cv2.dilate(res, dil_ker, iterations=1)
            self.er = np.where(self.er < .5, dil_res, self.er)
            
    def smallReg(self):
        lbl = label(self.er)
        num_reg = []
        for i in range(int(lbl.max()) + 1):
            if len(np.where(lbl == i)[0]) < self.m * self.n / 5000:
                num_reg.append(i)

        for nr in num_reg:
            self.er = np.where(lbl == nr, 0., self.er)

    def measureWidth(self):
        sk_idx = np.where(skeletonize(self.er) == 1)
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
                    wid_er.append((_w + 1) * 1)
                    break
                else:
                    _w += 1
        mu = sum(wid_er) / len(sel_idx)
        sig = np.std(wid_er)
        Z_45 = 1.65     # standard normal value for 90 %
        wid = Z_45 * sig / np.sqrt(tot_len // 10) + mu
        return wid

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

    def removeShorts(self, param_del=20):
        tol = np.sqrt(self.m**2 + self.n**2) / param_del
        lbl_sk = label(self.sk, background=0, connectivity=2)
        lbl_er = label(self.er, background=0, connectivity=2)
        for _ls in range(int(lbl_sk.max())):
            ls = _ls + 1
            if np.sum(lbl_sk == ls) < tol:
                idx = np.where(lbl_sk == ls)
                self.er = np.where(lbl_er == lbl_er[idx[0][0], idx[1][0]], 0., self.er)

    def removeHoles(self, param_sz=100):
        lbl = label(self.er, background=1, connectivity=1)
        del_tol = self.m * self.n / param_sz**2
        for lbl_i in range(1, np.max(lbl) + 1):
            idx_i = np.where(lbl == lbl_i)
            num_i = len(idx_i[0])
            if num_i < del_tol:
                self.er[idx_i] = 1

    def skeletonize(self):
        '''
        skeletonization of edge region
        '''
        rein = Reinitial()
        self.psi = rein.getSDF(.5 - self.er)
        gx, gy = self.imgrad(self.psi)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        
        self.sk_phi = np.where((ng < .85) * (self.er > .5), 1., 0.)
        self.sk = skeletonize(self.sk_phi)
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
        _gap = 1 / self.gap / 5
        pts = np.arange(0, -self.gap, -_gap)
        self.new_er = np.zeros_like(self.er)
        banned = np.zeros((len(self.ind_end), 1))
        
        _D = np.arange(self.num_pts)
        D = np.array([_D * _D, _D, np.ones_like(_D)]).T
        for k, pt in enumerate(pts[1:]):
            for iii, idx in enumerate(self.ind_end):
                if (len(idx) < self.maxlen_cv - 1) or banned[iii]:
                    continue

                b = np.array(list(zip(*idx[::self.gap + 1]))).T
                mDb = min(len(b), len(D))
                abc = np.linalg.lstsq(D[:mDb, :], b[:mDb, :], rcond=None)[0]
                abc[-1, :] = b[0]

                yy = np.round(abc[0, 0] * pt * pt + abc[1, 0] * pt + abc[2, 0]).astype(int)
                xx = np.round(abc[0, 1] * pt * pt + abc[1, 1] * pt + abc[2, 1]).astype(int)

                _yy = np.round(abc[0, 0] * pts[k] * pts[k] + abc[1, 0] * pts[k] + abc[2, 0]).astype(int)
                _xx = np.round(abc[0, 1] * pts[k] * pts[k] + abc[1, 1] * pts[k] + abc[2, 1]).astype(int)

                if (yy == _yy) and (xx == _xx):
                    continue

                if yy >= self.m or xx >= self.n:
                    banned[iii] = 1
                    continue
                if -yy >= self.m or -xx >= self.n:
                    banned[iii] = 1
                    continue
                elif self.psi[_yy, _xx] > self.psi[yy, xx]:
                # elif _ng[yy, xx] < .75 and _psi[yy, xx] > 0:
                    banned[iii] = 1
                    continue
                # elif ek[yy, xx] > .5 and psi[yy, xx] > 0:
                #     banned[iii] = 1
                #     continue
                self.new_er[yy, xx] = 1
                    
            # if np.abs(self.new_er).sum() > 0:
                # self.sk = np.where(self.new_er + self.sk > .5, 1., 0.)
                
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


if __name__ == '__main__':
    pass