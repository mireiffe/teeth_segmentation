# system libs
import os
from os.path import join, dirname, abspath
from configparser import ConfigParser, ExtendedInterpolation

# general libs
import cv2
import numpy as np
import matplotlib
matplotlib.use(backend='Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Pytorch
import torch
from torch.utils.data import DataLoader

# morphological imaging libs
from skimage.measure import label
from skimage.morphology import skeletonize

# custom libs
from gadf import GADF
import myTools as mts
from reinst import ThreeRegions
from reinitial import Reinitial

from _networks import model
from _networks.dataset import ErDataset


class PseudoER():
    dir_network = '/home/users/mireiffe/Documents/Python/ERLearning/results/'
    def __init__(self, args, num_img, scaling=False):
        self.num_img = num_img

        self.config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.config.optionxform = str
        self.config.read(args.path_cfg)

        self.config['DEFAULT']['HOME'] = abspath(join(dirname(abspath(__file__)), *[os.pardir]))

        # load configure file
        dir_ld = f"{self.dir_network}{self.config['DEFAULT']['dir_stdict']}"
        self.config.read(os.path.join(dir_ld, 'info_train.ini'))
        
        if args.device: 
            self.config['TRAIN'].update({'decvice': args.device[0], 'device_ids': args.device[1]})

        # set main device and devices
        cfg_train = self.config['TRAIN']
        self.dvc = cfg_train["device"]
        self.ids = cfg_train["device_ids"]
        self.lst_ids = [int(id) for id in self.ids]
        self.dvc_main = torch.device(f"{self.dvc}:{self.ids[0]}")
        self.scaling = scaling

    def getEr(self):
        net = self.setModel()
        net.eval()
        with torch.no_grad():
            _img, _er = self.inference(net)
        img = torch.Tensor.cpu(_img).squeeze().permute(1, 2, 0).numpy()
        er = torch.Tensor.cpu(_er).squeeze().numpy()
        return img, er

    def setModel(self):
        cfg_dft = self.config['DEFAULT']
        cfg_model = self.config['MODEL']

        kwargs = {
            k: eval(v) 
            for k, v in self.config.items(cfg_model['name'])
            if k not in cfg_dft.keys()
        }
        net = getattr(model, cfg_model['name'])(**kwargs).to(device=self.dvc_main)

        if self.dvc == 'cuda':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.lst_ids)[1:-1]

            net = torch.nn.DataParallel(net, device_ids=self.lst_ids)
            print(f"Using main device <{self.dvc_main}> and devices with <IDs: {self.ids}>")
        else:
            print(f"Using main device <{self.dvc_main}>")

        # Load parameters
        file_ld = os.path.join(self.dir_network, *[cfg_dft['dir_stdict'], f"checkpoints/{cfg_dft['num_cp']}.pth"])
        checkpoint = torch.load(file_ld, map_location='cpu')
        try:
            net.load_state_dict(checkpoint['net_state_dict'])
        except KeyError:
            net.load_state_dict(checkpoint['encoder_state_dict'])

        net.to(device=self.dvc_main)
        print(f'Model loaded from {file_ld}')
        return net

    def inference(self, net, dtype=torch.float):
        dir_dt = '/home/users/mireiffe/Documents/Python/TeethSeg/data/testimgs/'

        data_test = ErDataset(None, dir_dt, split=[[self.num_img, self.num_img+1]], wid_dil='auto', mode='test')
        loader_test = DataLoader(
            data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)

        for k, btchs in enumerate(loader_test):
            imgs = btchs[0].to(device=self.dvc_main, dtype=dtype)

            # _m, _n = imgs.shape[2:4]
            # image scaling
            # if (imgs.nelement() / imgs.shape[1] > 1200**2) and self.scaling:
                # imgs = TF.resize(imgs, 500, interpolation=InterpolationMode.NEAREST)
            # _imgs = TF.resize(imgs, 200, interpolation=InterpolationMode.BILINEAR)

            # preds = net(_imgs)
            preds = net(imgs)

            # preds = TF.resize(preds, _m, interpolation=InterpolationMode.BILINEAR)

            om, on = data_test.m, data_test.n
            m, n = imgs.shape[2:4]
            mi = (m - om) // 2
            ni = (n - on) // 2
        return imgs[..., mi:mi + om, ni:ni + on], preds[..., mi:mi + om, ni:ni + on]


class InitContour():
    def __init__(self, img:np.ndarray, per0:np.ndarray) -> np.ndarray:
        self.img = img
        self.per0 = per0
        self.m, self.n = self.per0.shape
        self.preset()
        self.per = mts.imDilErod(
            self.per, rad=max(round(self.wid_er / 1.5), 1),
            kernel_type='circular')

        # self.per = per0

        rein_all = Reinitial(width=None)
        self.rein_w5 = Reinitial(width=5, dim_stack=0)
        self.rein_w10 = Reinitial(width=10, dim_stack=2)

        # get landmarks
        phi_lmk = rein_all.getSDF(self.per - .5)
        lmk = self.getLandMarks(phi_lmk, area=5)
        self.phi_lmk = self.rein_w5.getSDF(.5 - (lmk[np.newaxis, ...] < 0))

        # bring them back
        self.phi_back = self.bringBack(self.phi_lmk, self.per, gap=7, dt=.3, mu=.5, nu=.1, reinterm=10, visterm=1, tol=2, max_iter=1500)

        plt.figure(); plt.imshow(self.per, 'gray')
        for ph in self.phi_back:
            plt.contour(ph, levels=[0], colors='lime', linewidths=3)
        #mts.savecfg('img0_back.png')

        # separate level sets 
        reg_sep = self.sepRegions(self.phi_back)
        phi_sep = self.rein_w5.getSDF(.5 - np.array(reg_sep))

        # initials
        _per = cv2.dilate(self.per, np.ones((3, 3)))
        phi_init = self.evolve(phi_sep, _per, dt=.3, mu=2, nu=.5, reinterm=3, visterm=3, tol=2, max_iter=200)
        # phi_init = self.evolve(phi=phi_sep, wall=self.per, dt=.3, mu=3, nu=.5, reinterm=3, visterm=3, tol=2, max_iter=200)

        self.phi0 = phi_init
        return

    def preset(self):
        self.wid_er = self.measureWidth()
        self.per = self.removeHoles(self.per0, param_sz=100)
        self.per = self.removeShorts(self.per, param_sz=100)

    def getLandMarks(self, phi0, area):
        m_phi = np.where(mts.local_minima(phi0), phi0, 0)
        phi = np.copy(phi0)
        while True:
            _lbl = label(phi < 0)
            _reg = np.zeros_like(_lbl)
            for l in np.unique(_lbl)[1:]:
                _r = np.where(_lbl == l, True, False)
                # if np.min(phi * _r) <= np.minimum(-area, np.min(m_phi*_r) / 2):
                if np.min(phi * _r) <= np.min(m_phi*_r) / 2:
                    _reg += _r
            if _reg.sum() == 0:
                break
            phi += _reg
        return phi

    def sepRegions(self, phi_back):
        lbl_per = label(self.per, background=1, connectivity=1)
        lbl_back = label(phi_back[0] < 0, background=0, connectivity=1)
        reg_sep = [np.zeros_like(phi_back[0]), ]
        for l in np.unique(lbl_per)[1:]:
            _backs = np.setdiff1d(np.unique(lbl_back * (lbl_per == l)), [0])
            if len(reg_sep) < len(_backs):
                for _ in range(len(_backs) - len(reg_sep)):
                    reg_sep.append(np.zeros_like(phi_back[0]))
            for _i, _l in enumerate(_backs):
                reg_sep[_i] += np.where(lbl_back == _l, 1., 0.)
        return reg_sep

    def evolve(self, phi, wall, dt, mu, nu, reinterm, visterm, tol, max_iter):
        phi0 = np.copy(phi)
        k = 0
        cmap = plt.cm.get_cmap('gist_rainbow', len(phi))
        dist = 1
        while True:
            regs = np.where(phi < dist, phi - dist, 0)
            all_regs = regs.sum(axis=0)
            Fc = (- (all_regs - regs) - 1)

            kapp = mts.kappa(phi0, stackdim=0)[0]
            phi = phi0 + dt * ( (2*Fc + mu * kapp) * (1 - wall) + (nu * wall) )

            if k % visterm == 0:
                plt.figure(1)
                plt.cla()
                plt.imshow(wall, 'gray')
                for i, ph in enumerate(phi):
                    plt.contour(ph, levels=[0], colors=[cmap(i)], linewidth=1.2)
                # if k % (2*visterm) == 0:
                #     mts.savecfg(f'ppt{k // (2*visterm):03d}.png')
                plt.title(f'iter = {k:d}')
                plt.pause(.1)

            if k % reinterm == 0:
                reg0 = np.where(phi0 < 0, 1, 0)
                reg = np.where(phi < 0, 1, 0)
                setmn = (reg0 + reg - 2 * reg0 * reg).sum()
                print(setmn)
                phi = self.rein_w5.getSDF(np.where(phi < 0, -1., 1.))
                if (setmn  < tol) or (k > max_iter):
                    break

            k += 1
            phi = mts.remove_pos_lvset(phi)[0]
            phi0 = phi
        return phi

    def bringBack(self, phi, per, gap, dt, mu, nu, reinterm, visterm, tol, max_iter):
        wall = mts.gaussfilt(cv2.dilate(per, np.ones((2*gap + 1, 2*gap + 1))), sig=1)
        lbl_per = label(per, background=1, connectivity=1)
        # lbl_per = label(np.where(phi[0] < 0, 1, 0), background=0, connectivity=1)
        for l in np.unique(lbl_per)[1:]:
            _r = np.where(lbl_per == l)
            if (wall > 0.01)[_r].sum() == len(_r[0]):
                wall[_r] = 0

        wall = np.where(phi < 0, 0, wall)

        phi0 = np.copy(phi)
        k = 0
        while True:
            kapp = mts.kappa(phi0, stackdim=0)[0]
            phi = phi0 + dt * ( (-1 + mu * kapp) * (1 - wall) + nu * wall)

            if k % visterm == 0:
                plt.figure(1)
                plt.cla()
                plt.imshow(self.per, 'gray')
                plt.contour(phi[0], levels=[0], colors='lime', linewidth=1.2)
                plt.title(f'iter = {k:d}')
                plt.pause(.1)

            if k % reinterm == 0:
                reg0 = np.where(phi0 < 0, 1, 0)
                reg = np.where(phi < 0, 1, 0)
                setmn = (reg0 + reg - 2 * reg0 * reg).sum()
                print(setmn)
                if (setmn  < tol) or (k > max_iter):
                    break
                # phi = self.rein_w5.getSDF(np.where(phi < 0.1, -1., 1.))
                phi = self.rein_w5.getSDF(np.where(phi < 0, -1., 1.))
            k += 1
            phi0 = phi

        return phi

    def removeShorts(self, img, param_sz):
        res = img
        sk = skeletonize(res)
        len_diag = np.sqrt(self.m**2 + self.n**2)
        tol_len = len_diag / param_sz

        lbl_er = label(res, background=0, connectivity=2)
        lbl_sk = lbl_er * sk
        for l in np.unique(lbl_sk):
            if l == 0: continue
            if np.sum(lbl_sk == l) < tol_len:
                res = np.where(lbl_er == l, 0, self.per)
            
        return res

    def removeHoles(self, img, param_sz=100):
        _er = img
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
        return _er

    def measureWidth(self):
        sk_idx = np.where(skeletonize(self.per0) == 1)
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
                _ptch = self.per0[y0:_y+_w+2, x0:_x+_w+2]
                if _ptch.sum() < _ptch.size:
                    wid_er.append(2*_w + 1)
                    break
                else:
                    _w += 1
        mu = sum(wid_er) / len(sel_idx)
        sig = np.std(wid_er)
        Z_45 = 1.65     # standard normal value for 90 %
        return Z_45 * sig / np.sqrt(tot_len // 10) + mu


class Snake():
    def __init__(self, img:np.ndarray, per:np.ndarray, phi0) -> None:
        self.img = img
        self.per = per
        self.m, self.n = self.per.shape

        # find GADF for gray img
        GF = GADF(self.img.mean(axis=2))
        self.fa = GF.Fa
        self.er = GF.Er

        self.wid_er = self.measureWidth(self.per)
        self.rein = Reinitial(dt=.2, width=5, tol=0.01, dim_stack=0)

        self.phi0 = self.sepRegions(phi0)

    def sepRegions(self, phi):
        _rein = Reinitial(dt=.2, width=self.wid_er * 3, tol=0.01, fmm=True)
        lbl_phi = np.zeros_like(phi[0])
        for ph in phi:
            _lbl = label(np.where(ph < 0, 1., 0.), background=0, connectivity=1)
            lbl_phi = np.where(_lbl > .5, _lbl + lbl_phi.max(), lbl_phi)
        phis = []
        for l in np.unique(lbl_phi)[1:]:
            _reg = np.where(lbl_phi == l, 1., 0.)
            phis.append(_rein.getSDF(.5 - _reg))

        return np.array(phis)

    def measureWidth(self, per):
        sk_idx = np.where(skeletonize(per) == 1)
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
                _ptch = per[y0:_y+_w+2, x0:_x+_w+2]
                if _ptch.sum() < _ptch.size:
                    wid_er.append(2*_w + 1)
                    break
                else:
                    _w += 1
        mu = sum(wid_er) / len(sel_idx)
        sig = np.std(wid_er)
        Z_45 = 1.65     # standard normal value for 90 %
        return Z_45 * sig / np.sqrt(tot_len // 10) + mu

    def snake(self):
        dt = 0.2
        mu = 3
        reinterm = 3

        n_phis = len(self.phi0)
        cmap = plt.cm.get_cmap('gist_rainbow', n_phis)

        teg = [ThreeRegions(self.img) for nph in range(n_phis)]

        phis = np.copy(self.phi0)

        stop_reg = np.ones_like(self.per)
        stop_reg[2:-2, 2:-2] = 0
        
        # self.use_er = self.er * self.per
        # self.use_er = cv2.dilate(self.er.astype(float), kernel=np.ones((3, 3)))
        self.use_er = self.er * ((phis > -1).sum(axis=0) == n_phis)
        oma = self.use_er
        omc = (1 - oma) * (1 - stop_reg)
        # oms = (self.per - oma) * (1 - stop_reg)
        oms = (1 - oma) * (1 - stop_reg)

        k = 0
        while True:
            k += 1
            if k % reinterm == 0:
                phis = self.rein.getSDF(np.where(phis < 0, -1., 1.))

            dist = 1
            regs = np.where(phis < dist, phis - dist, 0)
            all_regs = regs.sum(axis=0)
            Fc = (- (all_regs - regs) - 1)

            for i in range(n_phis):
                teg[i].setting(phis[i, ...])

            gx, gy = mts.imgrad(phis.transpose((1, 2, 0)))
            Fa = - (gx.transpose((2, 0, 1)) * self.fa[..., 1] + gy.transpose((2, 0, 1)) * self.fa[..., 0])
            _Fs = np.array([- tg.force() for tg in teg])
            Fs = mts.gaussfilt(_Fs, sig=1, stackdim=0)

            kap = mts.kappa(phis, stackdim=0)[0]
            F = 1*Fa*oma + Fs*oms + Fc*omc + mu*kap
            new_phis = phis + dt * F

            err = np.abs(new_phis - phis).sum() / new_phis.size
            if err < 1E-04 or k > 250:
            # if err < 1E-04 or k > 1:
                break
        
            if k in [1, 2] or k % 9 == 0:
                plt.figure(1)
                plt.cla()
                plt.imshow(self.img)
                # plt.imshow(self.per, mts.colorMapAlpha(plt), vmax=2)
                plt.imshow(oma, alpha=.8, cmap=mts.colorMapAlpha(plt))
                for i, ph in enumerate(new_phis):
                    _pr = np.where(ph > 0)
                    if len(_pr[0]) == self.m * self.n:
                        continue
                    plt.contour(ph, levels=[0], colors=[cmap(i)])
                plt.title(f'iter = {k:d}')
                # plt.show()
                plt.pause(.1)

            new_phis, teg = mts.remove_pos_lvset(new_phis, teg)
            n_phis = len(new_phis)
            phis = new_phis

        return new_phis


class IdRegion():
    def __init__(self, img:np.ndarray, phi_res) -> None:
        self.img = img
        self.phi_res = phi_res

        self.m, self.n = self.img.shape[:2]
        self.lbl_reg = self.setReg(self.phi_res)

        # stimg = np.where(img[..., 0] < 1E-04, 0, img[..., 1] / img[..., 0] -.35*img[..., 2])
        # sstimg = (stimg - stimg.min()) / (stimg.max() - stimg.min())
        # idx = np.where(sstimg >= 1E-04)
        # temp = cv2.threshold((sstimg[idx] * 255).astype('uint8'), -1, 255, cv2.THRESH_OTSU)
        # timg = np.zeros_like(stimg)
        # timg[idx] = temp[1][..., 0]

        # ttimg = cv2.adaptiveThreshold((sstimg * 255).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 0)

        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(ttimg, 'gray')
        # plt.contour(self.lbl_reg, levels=np.unique(self.lbl_reg)[1:], colors='lime', linewidths=3)
        # plt.show()

        
        self.res = self.regClass()

    def setReg(self, phis):
        res = -np.ones((self.m, self.n))
        for i, phi in enumerate(phis):
            if len(np.where(phi < 0)[0]) > self.m * self.n / 500:
                res = np.where(phi < 0, i, res)
        return res

    def regClass(self):
        lbl_inreg = self.removeBG(self.lbl_reg)
        lbl_kapp = self.regClassKapp(lbl_inreg)
        lbl_intia = self.regInertia(lbl_kapp)

        cand_rm = self.candVLine(lbl_intia)
        lbl_vl = self.regClassVLine(self.img, lbl_intia, cand_rm)
        lbl_sd = self.removeSide(self.img, lbl_vl)
        return lbl_sd

    @staticmethod
    def removeBG(lbl):
        tnb = np.unique(lbl[[0, -1], :])
        lnr = np.unique(lbl[:, [0, -1]])

        res = np.copy(lbl)
        for l in np.unique(np.union1d(tnb, lnr)):
            res = np.where(lbl == l, -1., res)
        return res

    def regClassKapp(self, lbl):
        Rein = Reinitial(dt=0.1, width=10, tol=0.01)

        bdr = np.zeros_like(lbl)
        bdr[3:-3, 3:-3] = 1

        reg_nkapp = []
        reg_kapp = {}
        for l in np.unique(lbl)[1:]:
            _reg = np.where(lbl == l, 1., 0.)
            if _reg.sum() < self.m*self.n / 300:
                reg_nkapp.append(l)
                continue

            _phi = Rein.getSDF(.5 - _reg)
            _kapp = mts.kappa(_phi, mode=0)[0]
            _kapp = mts.gaussfilt(_kapp, sig=.5)

            reg_cal = (np.abs(_phi) < 1.5) * bdr
            kapp_p = np.where(_kapp > 1E-04, 1, 0)
            kapp_n = np.where(_kapp < -1E-04, 1, 0)

            n_kapp_p = (kapp_p * reg_cal).sum()
            n_kapp_n = (kapp_n * reg_cal).sum()

            reg_kapp[l] = (n_kapp_p - n_kapp_n) / (reg_cal.sum())

        for rk, rv in reg_kapp.items():
            if rv < .2:
                reg_nkapp.append(rk)

        stimg = np.where(self.img[..., 0] < 1E-04, 0, self.img[..., 1] / self.img[..., 0])
        # mu_img = np.mean(self.img, where=np.where(self.img==0, False, True))
        # var_img = np.var(self.img, where=np.where(self.img==0, False, True))
        mu_img = np.mean(stimg, where=np.where(np.abs(stimg) < 1E-04, False, True))
        res = np.copy(lbl)
        for rnk in reg_nkapp:
            _mu_r = np.mean(stimg, where=(lbl == rnk))
            if _mu_r <= mu_img:
                res = np.where(lbl == rnk, -1., res)
        return res

    def regInertia(self, lbl):
        eig_lst = {}
        rat_lst = {}
        cenm_lst = {}
        for l in np.unique(lbl):

            if l < 0: continue
            r_idx = np.where(lbl == l)

            # y and x order
            cenm = np.sum(r_idx, axis=1) / len(r_idx[0])
            cen_idx = r_idx[0] - cenm[0], r_idx[1] - cenm[1]

            Ixx = np.sum(cen_idx[0]**2)
            Iyy = np.sum(cen_idx[1]**2)
            Ixy = -np.sum(cen_idx[0]*cen_idx[1])

            intiaT = [[Ixx, Ixy], [Ixy, Iyy]]
            D, Q = mts.sortEig(intiaT)

            eig_lst[l] = (D, Q)
            rat_lst[l] = D[0] / D[1]
            cenm_lst[l] = cenm

        mu_rat = np.mean(list(rat_lst.values()))
        var_rat = np.var(list(rat_lst.values()))

        # mu_img = np.mean(self.img, where=np.where(self.img==0, False, True))
        stimg = np.where(self.img[..., 0] < 1E-04, 0, self.img[..., 1] / self.img[..., 0])
        mu_img = np.mean(stimg, where=np.where(np.abs(stimg) < 1E-04, False, True))

        res = np.copy(lbl)
        for l in np.unique(lbl):
            if l < 0: continue
            # _mu_r = np.mean(self.img.transpose((2, 0, 1)), where=_reg)
            _mu_r = np.mean(stimg, where=(lbl == l))
            if _mu_r <= mu_img:
                _ang = np.abs(eig_lst[l][1][0, 1]) >= np.cos(np.pi / 4)
                if (rat_lst[l] > mu_rat - 0 * np.sqrt(var_rat)) and (_ang):
                    res = np.where(lbl == l, -1, res)

        return res

        import matplotlib.patheffects as PathEffects
        plt.figure()
        plt.imshow(self.img)
        for ph in self.phi_res:
            if (ph < 0).sum() >= self.m*self.n / 1000:
                plt.contour(ph, levels=[0], colors='lime')
        for l in np.unique(lbl):
            if (self.phi_res[int(l)] < 0).sum() < self.m*self.n / 1000:
                continue

            if l < 0: continue
            # plt.quiver(cenm_lst[l][1], cenm_lst[l][0], eig_lst[l][1][0, 0], eig_lst[l][1][1, 0], angles='xy', color='blue')
            # plt.quiver(cenm_lst[l][1], cenm_lst[l][0], eig_lst[l][1][0, 1], eig_lst[l][1][1, 1], angles='xy', color='blue')
            # tt = f'rat: {eig_lst[l][0][0] / eig_lst[l][0][1]:.2f}\nang: {np.arccos(np.abs(eig_lst[l][1][0, 1]))*180/np.pi:.2f}'
            # txt = plt.text(cenm_lst[l][1], cenm_lst[l][0], tt, color='black', fontsize=9)
            # txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            tt1 = f'rat: {eig_lst[l][0][0] / eig_lst[l][0][1]:.2f}'
            tt2 = f'ang: {np.arccos(np.abs(eig_lst[l][1][0, 1]))*180/np.pi:.1f}'
            if rat_lst[l] >= mu_rat:
                txt1 = plt.text(cenm_lst[l][1], cenm_lst[l][0], tt1, color='r', fontsize=9)
            else:
                txt1 = plt.text(cenm_lst[l][1], cenm_lst[l][0], tt1, color='black', fontsize=9)
            if np.arccos(np.abs(eig_lst[l][1][0, 1]))*180/np.pi <  20:
                txt2 = plt.text(cenm_lst[l][1], cenm_lst[l][0]+10, tt2, color='r', fontsize=9)
            else:
                txt2 = plt.text(cenm_lst[l][1], cenm_lst[l][0]+10, tt2, color='black', fontsize=9)
            txt1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            txt2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            if (rat_lst[l] >= mu_rat) or (np.arccos(np.abs(eig_lst[l][1][0, 1]))*180/np.pi <  20):
                _mu_r = np.mean(self.img.transpose((2, 0, 1)), where=self.phi_res[int(l)] < 0)
                if _mu_r <= mu_img:
                    plt.imshow(self.phi_res[int(l)] < 0, vmax=2, cmap=mts.colorMapAlpha(plt))
        plt.title(f'Average = {mu_rat:.2f}')

    @staticmethod
    def candVLine(lbl):
        reg2bchck = []
        thres = .80
        for l in np.unique(lbl):
            if l < 0: continue
            _reg = np.where(lbl == l)

            # _x = np.unique(_reg[1])
            # n_samples = max(round(len(_x) / 2.), 1)
            _x = _reg[1]
            # n_samples = max(round(len(_x) / 20.), 1)
            n_samples = len(_x)
            
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
        _init_k = np.array([[100, 0, 0], lab[m_a[0], m_a[1], :], [50, 0, lab[m_b[0], m_b[1], 2]]])

        for l in cand:
            if l < 0: continue
            _reg = np.where(lbl == l)

            # _x = np.unique(_reg[1])
            # n_samples = max(round(len(_x) / 2.), 1)
            _x = _reg[1]
            n_samples = max(round(len(_x) / 10.), 1)
            
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
                # p_lbl = vl_lbl
                # p_img = vl_img
                # p_lab = vl_lab

                p_m_a = np.unravel_index(np.argmax(p_lab[..., 1]), p_lab[..., 1].shape)
                p_m_b = np.unravel_index(np.argmax(p_lab[..., 2]), p_lab[..., 2].shape)
                p_init_k = np.array([[100, 0, 0], [50, p_lab[p_m_a[0], 1], p_lab[p_m_a[0], 2]], [50, 0, p_lab[p_m_b[0], 2]]])
                init_k = (_init_k + p_init_k) / 2
                kmeans = KMeans(n_clusters=3, init=init_k).fit(p_lab)
                kmlbl = kmeans.labels_

                l_kmlbl = kmlbl[p_lbl == l]
                # modes_reg.append(int(stats.mode(l_kmlbl)[0]))
                # modes_reg.append(l_kmlbl)
                modes_reg += list(l_kmlbl)

                if 1 == 0:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d') # Axe3D object
                    ax.scatter(p_lab[..., 1], p_lab[..., 2], p_lab[..., 0], vmin=-1, vmax=33, c=p_lbl, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
                    ax.scatter(init_k[..., 1], init_k[..., 2], init_k[..., 0], marker='*', vmax=2, c=[0, 1, 2], s= 60, alpha=1, cmap=mts.colorMapAlpha(plt))
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d') # Axe3D object
                    ax.scatter(p_lab[..., 1], p_lab[..., 2], p_lab[..., 0], vmax=2, c=kmlbl, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
                    ax.scatter(init_k[..., 1], init_k[..., 2], init_k[..., 0], marker='*', vmax=2, c=[0, 1, 2], s= 60, alpha=1, cmap=mts.colorMapAlpha(plt))
            
            if int(stats.mode(modes_reg)[0]) == 1:
                res = np.where(res == l, -1, res)

        ### meeting
        idx = []
        for l in [4, 5, 10, 17, 25]:
            idx.append(np.where(lbl == l))
        
        col = []
        for ii in idx:
            col.append(lab[ii])

        p_lab = np.concatenate(col, axis=0)

        p_m_a = np.unravel_index(np.argmax(p_lab[..., 1]), p_lab[..., 1].shape)
        p_m_b = np.unravel_index(np.argmax(p_lab[..., 2]), p_lab[..., 2].shape)
        p_init_k = np.array([[100, 0, 0], [50, p_lab[p_m_a[0], 1], p_lab[p_m_a[0], 2]], [50, 0, p_lab[p_m_b[0], 2]]])
        init_k = (_init_k + p_init_k) / 2
        kmeans = KMeans(n_clusters=3, init=init_k).fit(p_lab)
        kmlbl = kmeans.labels_

        lens = []
        for cc in col:
            lens.append(len(cc))

        km = []
        clens = 0
        for lns in lens:
            km.append(kmlbl[clens:clens+lns])
            clens += lns

        temp3 = np.zeros_like(lbl)
        for ii, id in enumerate(idx):
            temp3[id] = km[ii] + 1

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


class ReinitialKapp(Reinitial):
    def __init__(self, dt:float=0.1, width:float=5, tol:float=1E-02, iter:int=None, dim:int=2, debug=False, fmm=False, mu=0.1) -> np.ndarray:
        super().__init__(dt=dt, width=width, tol=tol, iter=iter, dim=dim, debug=debug, fmm=fmm)
        self.mu = mu

    def update(self, phi):
        m, n = phi.shape[:2]

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
        kapp = mts.gaussfilt(mts.kappa(phi)[0], sig=np.ceil(m*n/300000))
        # kapp = mts.kappa(phi)[0]
        _phi = phi - self.dt * (_sign0 * _G - self.mu * kapp)
        return _phi
