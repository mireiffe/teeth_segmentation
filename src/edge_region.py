import os
from os.path import join, dirname, abspath
from configparser import ConfigParser, ExtendedInterpolation

import torch

from _networks.dataset import ErDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# custom libs
from _networks import model


class EdgeRegion():
    def __init__(self, path_cfg, num_img):
        self.num_img = num_img

        self.config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.config.optionxform = str
        self.config.read(path_cfg)

        self.config['DEFAULT']['HOME'] = abspath(join(dirname(abspath(__file__)), *[os.pardir]))

        # set main device and devices
        cfg_train = self.config['TRAIN']
        self.dvc = cfg_train["device"]
        self.ids = cfg_train["device_ids"]
        self.lst_ids = [int(id) for id in self.ids]
        self.dvc_main = torch.device(f"{self.dvc}:{self.ids[0]}")

    def getEr(self):
        net = self.setModel()
        net.eval()
        with torch.no_grad():
            _er = self.inference(net)
        er = torch.Tensor.cpu(_er).squeeze().numpy()
        return er

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
        file_ld = os.path.join(cfg_dft['dir_stdict'], f"checkpoints/{cfg_dft['num_cp']}.pth")
        checkpoint = torch.load(file_ld, map_location='cpu')
        try:
            net.load_state_dict(checkpoint['net_state_dict'])
        except KeyError:
            net.load_state_dict(checkpoint['encoder_state_dict'])

        net.to(device=self.dvc_main)
        print(f'Model loaded from {file_ld}')
        return net

    def inference(self, net, dtype=torch.float):
        cfg_dt = self.config['DATASET']

        if cfg_dt['name'] in {'er_labeled', 'er_less'}:
            data_test = ErDataset(cfg_dt['path'], split=self.num_img, wid_dil='auto')
        else:
            raise NotImplementedError('There are no such dataset')
        n_test = len(data_test)
        loader_test = DataLoader(
            data_test, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)

        for k, btchs in enumerate(loader_test):
            imgs = btchs[0].to(device=self.dvc_main, dtype=dtype)
            preds = net(imgs)

            om, on = data_test.m, data_test.n
            m, n = imgs.shape[2:4]
            mi = (m - om) // 2
            ni = (n - on) // 2
        return preds[..., mi:mi + om, ni:ni + on]


if __name__=='__main__':
    _er = EdgeRegion('/home/users/mireiffe/Documents/Python/TeethSeg/cfg/inference.ini', 51)
    er = _er.getEr()

    

    plt.imshow(er > .5, 'gray')
    plt.show()