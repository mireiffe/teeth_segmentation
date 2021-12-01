import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from reinitial import Reinitial
import myTools as mts


class IdRegion():
    eps = np.finfo(float).eps
    jet_alpha = mts.colorMapAlpha(plt)
    
    def __init__(self, dict, dir_img) -> None:
        self.dir_img = dir_img
        self.dict = dict
        self.img:np.ndarray = dict['img']
        self.phi_res = dict['phi_res']

        self.m, self.n = self.img.shape[:2]

        self.lbl_reg = self.setReg(self.phi_res)
        self.res = self.regClass()
        self._saveSteps()

    def setReg(self, phis):
        res = -np.ones((self.m, self.n))
        for i, phi in enumerate(phis):
            if len(np.where(phi < 0)[0]) > self.m * self.n / 750:
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
        for l in np.unique(lbl):
            if l < 0: continue
            _reg = np.where(lbl == l, 1., 0.)
            if _reg.sum() < self.m*self.n / 1000:
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
            if reg_kapp[l] < .1:
                reg_nkapp.append(l)

        stimg = (self.img[..., 1] + self.img[..., 2]) - 2*self.img[..., 0]
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
        stimg = (self.img[..., 1] + self.img[..., 2]) - 2*self.img[..., 0]
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
        _init_k = np.array([[100, 0, 0], lab[m_a[0], m_a[1], :], [50, 0, lab[m_b[0], m_b[1], 2]]])

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

                # p_lbl = vl_lbl[vl_lbl >= 0]
                # p_img = vl_img[vl_lbl >= 0]
                # p_lab = vl_lab[vl_lbl >= 0]
                p_lbl = vl_lbl
                p_img = vl_img
                p_lab = vl_lab

                p_m_a = np.unravel_index(np.argmax(p_lab[..., 1]), p_lab[..., 1].shape)
                p_m_b = np.unravel_index(np.argmax(p_lab[..., 2]), p_lab[..., 2].shape)
                p_init_k = np.array([[100, 0, 0], p_lab[p_m_a[0], :], [50, 0, p_lab[p_m_b[0], 2]]])
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
                    ax.scatter(p_lab[..., 1], p_lab[..., 2], p_lab[..., 0], vmin=-1, vmax=13, c=p_lbl, s= 20, alpha=0.5, cmap=mts.colorMapAlpha(plt))
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

        ax.imshow(obj)
        if face is not None:
            _res = np.where(face < 0, 0, face)
            plt.imshow(_res, alpha=.4, cmap='rainbow_alpha')
        if contour is not None:
            Rein = Reinitial(dt=.1)
            ReinKapp = ReinitialKapp(iter=10, mu=.1)
            clrs = ['lime'] * 100
            for i in range(int(np.max(contour))):
                _reg = np.where(contour == i+1, -1., 1.)
                for _i in range(5):
                    _reg = Rein.getSDF(_reg)
                    _reg = ReinKapp.getSDF(_reg)
                plt.contour(_reg, levels=[0], colors=clrs[i], linewidths=1)

        plt.axis('off')
        plt.savefig(f'{self.dir_img}{name}', dpi=1024, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _saveSteps(self):
        self._showSaveMax(self.lbl_reg, 'lbl_reg.pdf')
        self._showSaveMax(self.res, 'res.pdf')
        self._showSaveMax(self.img, 'res_c.pdf', contour=self.res)

        self.dict['lbl_reg'] = self.lbl_reg
        self.dict['res'] = self.res

        pass


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
