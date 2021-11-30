import cv2
import numpy as np

from skimage.measure import label


class GADF():
    eps = np.finfo(float).eps

    def __init__(self, img, sig=None, epsilon=1):
        self.img_orig = img
        self.epsilon = epsilon
        self.w, self.h = img.shape[:2]
        if len(img.shape) > 2:
            self.c = img.shape[2]
        else:
            self.c = 1

        if sig == None:
            sig = np.sqrt(self.w**2 + self.h**2) / 300  
        img = self.gaussfilt(self.img_orig, sig=sig)
        self.Fa = self.gadf(img)
        er = self.edgeRegion()
        er1 = self.fineEr(iter=5, coeff=4)
        er2 = self.fineEr(er1, iter=2, coeff=4)
        # res = self.dilation(er2, len=2)
        # res = self.fineEr(er2, iter=2, coeff=4)

        self.Er = er2

    def fineEr(self, er=None, iter=4, coeff=4):
        if er is not None:
            ler, ser = self.smallRegion(er, iter=iter, coeff=coeff)
        else:
            ler, ser = self.smallRegion(self.Er, iter=iter, coeff=coeff)
        del_s = self.delEr(ser)
        er = ler + del_s
        return er

    def delEr(self, er):
        gimg = self.img_orig.mean(axis=2)
        _lbl = label(er, background=0,connectivity=1)
        
        N = {}
        Sig = {}
        for i in range(1, _lbl.max() + 1):
            N[i] = np.sum(_lbl == i)
            gimg_reg = (_lbl == i) * gimg
            Sig[i] = ((gimg_reg ** 2).sum() - gimg.sum()) / N[i]

        lst_N, lst_Sig = list(N.values()), list(Sig.values())
        mu_N, sig_N = np.mean(lst_N), np.std(lst_N)
        mu_Sig, sig_Sig = np.mean(lst_Sig), np.std(lst_Sig)

        ker = np.ones((3, 3))
        mean_loc = cv2.filter2D(gimg, -1, ker, borderType=cv2.BORDER_REFLECT)
        Sig_loc = np.sqrt(cv2.filter2D((gimg - mean_loc) ** 2, -1, ker, borderType=cv2.BORDER_REFLECT))

        dist_sig = (Sig_loc - mu_Sig) / (sig_Sig + self.eps)

        fun_alpha = lambda x: 1 / (1 + np.exp(-x) + self.eps)
        nx = mu_N + fun_alpha(-dist_sig) * sig_N
        for k, nn in N.items():
            if np.sum((nx > nn) * (_lbl == k)) > .5:
                er = np.where(_lbl == k, 0, er)
        return er

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
    def smallRegion(er, iter=5, coeff=4) -> tuple:
        lbl = label(er, background=0,connectivity=1)
        sz_reg = {}
        for i in range(1, lbl.max() + 1):
            sz_reg[i] = np.sum(lbl == i)
        _lst = list(sz_reg.values())
        _mu = np.mean(_lst)
        _sig = np.std(_lst)

        cnt = 0
        while True:
            cnt += 1
            lim_v = _mu + coeff * _sig
            _items = list(sz_reg.items())
            for k, sr in _items:
                if sr > lim_v: del sz_reg[k]

            if cnt > 3: break
            
            _lst = list(sz_reg.values())
            _mu = np.mean(_lst)
            _sig = np.std(_lst)
        
        part_small = np.zeros_like(lbl)
        for k in sz_reg.keys():
            part_small += (lbl == k)
        part_large = er - part_small
        return part_large, part_small


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
