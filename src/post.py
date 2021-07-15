from os.path import join

import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label
import skfmm
from sklearn.cluster import KMeans

from reinitial import Reinitial


class PostProc():
    eps = np.finfo(float).eps
    
    def __init__(self, dict, dir_img) -> None:
        self.dir_img = dir_img

        self.img = dict['img']
        self.seg_er = dict['seg_er']
        self.er = dict['er']
        self.phi = dict['phi'][..., 0]
        self.m, self.n = self.er.shape

        self.GADF = GADF(self.img)
        self.Fa = self.GADF.Fa
        self.er_Fa = self.GADF.Er

        # FOR DEBUG =================
        def _show():
            X, Y = np.mgrid[0:self.m, 0:self.n]
            plt.figure()
            plt.imshow(self.img)
            plt.imshow(self.lbl0, alpha=.3)
            plt.quiver(Y, X, self.Fa[..., 0], self.Fa[..., 1], angles='xy', scale_units='xy', scale=1, color='blue')
            plt.show()
        # =============================

        self.lbl0 = self.labeling()


        self.soaking()
        self.lbl = self.labeling()
        self.lbl_fa = self.toGADF(self.lbl)
        self.tot_lbl = self.zeroReg(self.lbl_fa)
        
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('gist_rainbow')(range(ncolors))
        # change alpha values
        color_array[:,-1] = np.linspace(0.0, 1.0, ncolors)
        # create a colormap object
        from matplotlib.colors import LinearSegmentedColormap
        map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)
        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)

        plt.figure()
        plt.imshow(self.img)
        plt.imshow(self.lbl_fa, alpha=.4, cmap='rainbow_alpha')
        for i in range(int(np.max(self.lbl_fa))):
            plt.contour(np.where(self.lbl_fa == i+1, -1., 1.), levels=[0], colors='r')
        plt.title('lbl fa')

        plt.figure()
        plt.imshow(self.img)
        plt.imshow(self.tot_lbl, alpha=.4, cmap='rainbow_alpha')
        for i in range(int(np.max(self.tot_lbl))):
            plt.contour(np.where(self.tot_lbl == i+1, -1., 1.), levels=[0], colors='r')
        plt.title('tot lbl')
        plt.show()

        # self.distSize()
        self.res = self.regClass(self.tot_lbl)
        self._saveSteps()

    def toGADF(self, lbl):
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        dt = 0.1
        # mu = .1
        mu = 0
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
        _phis = Rein.getSDF(np.transpose(_regs, (1, 2, 0)))

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
            
            gx, gy = self.imgrad(_phis)
            _Fa = - 1 * (gx.transpose((2, 0, 1)) * self.Fa[..., 0] + gy.transpose((2, 0, 1)) * self.Fa[..., 1]) * self.er_Fa * (self.lbl == 0)
            _Fb = - 1 * (1 - self.er_Fa)

            kap = self.kappa(_phis)[0] * (np.abs(_phis) < 5)
            _F = (_Fa + _Fb) * cal_regs + _Fo + mu * kap.transpose((2, 0, 1))
            new_phis = _phis + dt * _F.transpose((1, 2, 0))

            err = np.abs(new_phis - _phis).sum() / new_phis.size
            if err < 1E-04 or _k > 3:
                break
        
            if _k % 3 == 1:
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

    def labeling(self):
        '''
        labeling connected region (0 value for not assigned region)
        '''
        seg_res = np.where(self.phi < 0, 1., 0.)
        lbl = label(seg_res, background=0, connectivity=1)
        del_tol = self.m * self.n / 750
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
                ptch_img = self.img[idx[0]-_k:idx[0]+_k+1, idx[1]-_k:idx[1]+_k+1, :]
                ptch = lbl[idx[0]-_k:idx[0]+_k+1, idx[1]-_k:idx[1]+_k+1]
                ele_ptch = np.unique(ptch)
                if len(ele_ptch) > 2:
                    break
            min_dist = []
            for ep in ele_ptch[1:]:
                idx_ep = np.where(ptch == ep)
                l2dist = np.sqrt(((val_img - ptch_img[idx_ep])**2).sum(axis=-1))
                min_dist.append(l2dist.min())
            el_min = np.argmin(min_dist) + 1
            _lbl[idx] = ele_ptch[el_min]
                    
        return lbl + _lbl

    def regClass(self, lbl):
        num_reg = int(lbl.max())

        indic_kapp = {}
        for ir in range(num_reg):
            if (ir + 1) not in lbl:
                continue
            _reg = np.where(lbl == (ir+1), -1., 1.)
            # _phi = Rein.getSDF(_reg)
            _phi = skfmm.distance(_reg)
            _kapp = self.kappa(_phi, mode=0)[0]
            _kapp = self.gaussfilt(_kapp, sig=2)

            cal_reg = np.abs(_phi) < 2
            p_kapp = np.where(_kapp > 0, _kapp, 0)
            n_kapp = np.where(_kapp < 0, _kapp, 0)

            n_pkapp = ((_kapp > 0) * cal_reg).sum()
            n_nkapp = ((_kapp < 0) * cal_reg).sum()

            if n_pkapp < n_nkapp:
                indic_kapp[ir + 1] = n_pkapp - n_nkapp

        # for i, ind in indic_kapp.items():
        #     new_lbl = np.where(new_lbl == i, -1, new_lbl)

        # # second phase
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(self.img.reshape((-1, 3)))
        # kmlbl = kmeans.labels_.reshape((self.m, self.n))

        # km0 = ((kmlbl == 0) * self.img.mean(axis=2)).sum() / (kmlbl == 0).sum()
        # km1 = ((kmlbl == 1) * self.img.mean(axis=2)).sum() / (kmlbl == 1).sum()

        # mustBT = np.argmax([km0, km1])

        # indic_kmeans = {}
        # for ir in range(num_reg):
        #     if (ir + 1) not in lbl:
        #         continue
        #     _reg = np.where(lbl == (ir+1), 1., 0.)
        #     _indic = _reg * kmlbl if mustBT else _reg * (1 - kmlbl)
        #     indic_kmeans[ir+1] = _indic.sum() / _reg.sum()

        temp = lbl
        temp2 = lbl
        new_lbl = lbl
        for i, ind in indic_kapp.items():
            temp = np.where(temp == i, ind, temp)
            temp2 = np.where(temp2 == i, indic_kapp[i], temp2)
            new_lbl = np.where(new_lbl == i, -1, new_lbl)

        plt.figure()
        plt.imshow(temp)
        plt.savefig(f'{self.dir_img}debug_post.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.show()

        return new_lbl

    def distSize(self):
        '''
        distribution for size of region
        '''
        num_reg = np.max(self.tot_lbl)
        sz_reg = [np.sum(self.tot_lbl == (i + 1)) for i in range(num_reg)]

        self.mu_sz = sum(sz_reg) / num_reg
        mu_sz_2 = sum([sr ** 2 for sr in sz_reg]) / num_reg
        self.sig_sz = np.sqrt(mu_sz_2 - self.mu_sz ** 2)

    def _saveSteps(self):
        plt.figure()
        plt.imshow(self.lbl0)
        plt.savefig(f'{self.dir_img}lbl0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.lbl)
        plt.savefig(f'{self.dir_img}lbl1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.tot_lbl)
        plt.savefig(f'{self.dir_img}lbl2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.res)
        plt.savefig(f'{self.dir_img}lbl3.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.figure()
        plt.imshow(self.img)
        clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        clrs = ['r'] * 100
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors=clrs[i])
        plt.savefig(f'{self.dir_img}res_0.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        plt.close('all')
        plt.figure()
        plt.imshow(self.img)
        _res = np.where(self.res == -1, 0, self.res)
        plt.imshow(_res, alpha=.5, cmap='rainbow_alpha')
        plt.savefig(f'{self.dir_img}res_1.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors='r')
        plt.savefig(f'{self.dir_img}res_2.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')

        plt.close('all')
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.lbl0)
        plt.subplot(2, 2, 2)
        plt.imshow(self.tot_lbl)
        plt.subplot(2, 2, 3)
        plt.imshow(self.res)
        plt.subplot(2, 2, 4)
        plt.imshow(self.img)
        # clrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10
        clrs = ['r'] * 100
        for i in range(np.max(self.res)):
            plt.contour(np.where(self.res == i+1, -1., 1.), levels=[0], colors=clrs[i])
        plt.savefig(f'{self.dir_img}res_tot.png', dpi=200, bbox_inches='tight', facecolor='#eeeeee')
        # plt.pause(10)


    def kappa(self, phis, ksz=1, h=1, mode=0):
        x, y = self.imgrad(phis)
        if mode == 0:
            ng = np.sqrt(x**2 + y**2 + self.eps)
            nx, ny = x / ng, y / ng
            xx, _ = self.imgrad(nx)
            _, yy = self.imgrad(ny)
            return xx + yy, x, y, ng
        elif mode == 1:
            xx, yy, xy = self.imgrad(phis, order=2)
            res_den = xx * y * y - 2 * x * y * xy + yy * x * x
            res_num = np.power(x ** 2 + y ** 2, 1.5)
            ng = np.sqrt(x**2 + y**2 + self.eps)        # just for output
            return res_den / (res_num + self.eps), x, y, ng

    @staticmethod
    def imgrad(img: np.ndarray, order=1, h=1) -> np.ndarray:
        '''
        central difference
        '''
        nd = img.ndim
        if nd < 3:
            img = np.expand_dims(img, axis=-1)
        if order == 1:
            _x_ = img[:, 2:, ...] - img[:, :-2, ...]
            x_ = img[:, 1:2, ...] - img[:, :1, ...]
            _x = img[:, -1:, ...] - img[:, -2:-1, ...]

            _y_ = img[2:, :, ...] - img[:-2, :, ...]
            y_ = img[1:2, :, ...] - img[:1, :, ...]
            _y = img[-1:, :, ...] - img[-2:-1, :, ...]

            gx = np.concatenate((x_, _x_, _x), axis=1)
            gy = np.concatenate((y_, _y_, _y), axis=0)
            if nd < 3:
                gx = gx[..., 0]
                gy = gy[..., 0]
            return gx / (2 * h), gy / (2 * h)
        elif order == 2:
            _img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='symmetric')

            gxx = _img[1:-1, 2:, ...] + _img[1:-1, :-2, ...] - 2 * _img[1:-1, 1:-1, ...]
            gyy = _img[2:, 1:-1, ...] + _img[:-2, 1:-1, ...] - 2 * _img[1:-1, 1:-1, ...]
            gxy = _img[2:, 2:, ...] + _img[:-2, :-2, ...] - _img[2:, :-2, ...] - _img[:-2, 2:, ...]
            if nd < 3:
                gxx = gxx[..., 0]
                gyy = gyy[..., 0]
                gxy = gxy[..., 0]
            return gxx / (h * h), gyy / (h * h), gxy / (4 * h * h)

    @staticmethod
    def loadFile(path):
        with open(path, 'rb') as f:
            _dt = pickle.load(f)
        return _dt

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)


class GADF():
    eps = np.finfo(float).eps

    def __init__(self, img, sig=2, epsilon=1):
        self.img_orig = img
        self.epsilon = epsilon
        self.w, self.h = img.shape[:2]
        img = self.gaussfilt(self.img_orig, sig=sig)
        if len(img.shape) > 2:
            self.c = img.shape[2]
        else:
            self.c = 1

        self.Fa = self.gadf(img)
        self.Er = self.edgeRegion()

    def gadf(self, img) -> None:
        if self.c == 1:
            ngx, ngy = self.normalGrad(img)

            Ip = self.directInterp(img, (ngx, ngy), self.epsilon)
            In = self.directInterp(img, (ngx, ngy), -self.epsilon)

            coeff = np.sign(Ip + In - 2 * img)
            Fa = np.stack((coeff * ngx, coeff * ngy), axis=2)
        elif self.c == 3:
            h = 1E-02
            E = self.structTensor(img)
            Q = self.eigvecSort(E)
            v = Q[..., 0]

            num_part = 21
            xp = np.linspace(0, self.epsilon, num_part)
            xn = np.linspace(-self.epsilon, 0, num_part)
            yp, yn = [], []
            for p, n in zip(*[xp, xn]):
                yp.append(self.dux(img, v, p, h))
                yn.append(self.dux(img, v, n, h))
            
            lx = np.trapz(yp, dx=1 / 20, axis=0) - np.trapz(yn, dx=1 / 20, axis=0)

            Fa = np.sign(lx)[..., None] * v
        else:
            raise NotImplemented('Number of image channels is not 1 or 3.')
        return Fa

    def normalGrad(self, img) -> np.ndarray:
        gx, gy = self.imgrad(img)
        ng = np.sqrt(gx ** 2 + gy ** 2)
        return gx / (ng + self.eps), gy / (ng + self.eps)

    def structTensor(self, img):
        gx, gy = self.imgrad(img)
        Ei = np.array([[gx * gx, gx * gy], [gy * gx, gy * gy]])
        E = Ei.sum(axis=4).transpose((2, 3, 0, 1))
        return E

    def edgeRegion(self) -> None:
        F_ = np.stack((self.directInterp(self.Fa[..., 0], (self.Fa[..., 0], self.Fa[..., 1])),
            self.directInterp(self.Fa[..., 1], (self.Fa[..., 0], self.Fa[..., 1]))), axis=2)
        indic = np.sum(self.Fa * F_, axis=2)
        self.Er = np.where(indic < 0, 1, 0)
        return self.Er

    def dux(self, img, v, mag, h):
        '''
        input
        -----
        v: direction \n
        s: maginitude which is coefficient of v \n
        h: increment for finite differential \n
        '''
        _d = v.transpose((2, 0, 1))
        up = np.array([self.directInterp(img[..., i], _d, mag + h) 
            for i in range(self.c)])
        un = np.array([self.directInterp(img[..., i], _d, mag - h) 
            for i in range(self.c)])
        res = np.sqrt(np.sum(((up - un) / (2 * h)) ** 2, axis=0))
        return res

    @staticmethod
    def eigvecSort(E:np.ndarray) -> tuple:
        v, Q = np.linalg.eig(E)
        _idx = np.argsort(v, axis=-1)[..., ::-1]
        Q_idx = np.stack((_idx, _idx), axis=2)
        sorted_Q = np.take_along_axis(Q, Q_idx, axis=-1)
        return sorted_Q

    @staticmethod
    def imgrad(img: np.ndarray, order=1, h=1) -> np.ndarray:
        '''
        central difference
        '''
        nd = img.ndim
        if nd < 3:
            img = np.expand_dims(img, axis=-1)
        if order == 1:
            _x_ = img[:, 2:, ...] - img[:, :-2, ...]
            x_ = img[:, 1:2, ...] - img[:, :1, ...]
            _x = img[:, -1:, ...] - img[:, -2:-1, ...]

            _y_ = img[2:, :, ...] - img[:-2, :, ...]
            y_ = img[1:2, :, ...] - img[:1, :, ...]
            _y = img[-1:, :, ...] - img[-2:-1, :, ...]

            gx = np.concatenate((x_, _x_, _x), axis=1)
            gy = np.concatenate((y_, _y_, _y), axis=0)
            if nd < 3:
                gx = gx[..., 0]
                gy = gy[..., 0]
            return gx / (2 * h), gy / (2 * h)
        elif order == 2:
            _img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='symmetric')

            gxx = _img[1:-1, 2:, ...] + _img[1:-1, :-2, ...] - 2 * _img[1:-1, 1:-1, ...]
            gyy = _img[2:, 1:-1, ...] + _img[:-2, 1:-1, ...] - 2 * _img[1:-1, 1:-1, ...]
            gxy = _img[2:, 2:, ...] + _img[:-2, :-2, ...] - _img[2:, :-2, ...] - _img[:-2, 2:, ...]
            if nd < 3:
                gxx = gxx[..., 0]
                gyy = gyy[..., 0]
                gxy = gxy[..., 0]
            return gxx / (h * h), gyy / (h * h), gxy / (4 * h * h)

    @staticmethod
    def gaussfilt(img, sig=2, ksz=None, bordertype=cv2.BORDER_REFLECT):
        if ksz is None:
            ksz = ((2 * np.ceil(2 * sig) + 1).astype(int), (2 * np.ceil(2 * sig) + 1).astype(int))
        return cv2.GaussianBlur(img, ksz, sig, borderType=bordertype)

    @staticmethod
    def directInterp(img: np.ndarray, direct:tuple or list, mag=1) -> np.ndarray:
        m, n = img.shape[:2]
        y, x = np.indices((m, n))

        x_ = x + mag * direct[0]
        y_ = y + mag * direct[1]

        x_ = np.where(x_ < 0, 0, x_)
        x_ = np.where(x_ > n - 1, n - 1, x_)
        y_ = np.where(y_ < 0, 0, y_)
        y_ = np.where(y_ > m - 1, m - 1, y_)

        x1 = np.floor(x_).astype(int)
        x2 = np.ceil(x_).astype(int)
        y1 = np.floor(y_).astype(int)
        y2 = np.ceil(y_).astype(int)

        I1 = img[y1, x1, ...]
        I2 = img[y1, x2, ...]
        I3 = img[y2, x2, ...]
        I4 = img[y2, x1, ...]

        I14 = (y_ - y1) * I4 + (y2 - y_) * I1
        I23 = (y_ - y1) * I3 + (y2 - y_) * I2

        return (x_ - x1) * I23 +(x2 - x_) * I14