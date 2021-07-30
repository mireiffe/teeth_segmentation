import os
from os.path import join, dirname, abspath
from configparser import ConfigParser, ExtendedInterpolation

import torch

from dataset import ErDataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

import matplotlib.pyplot as plt

# custom libs
import model


dir_network = '/home/users/mireiffe/Documents/Python/ERLearning/results/'


class EdgeRegion():
    def __init__(self, args, num_img, scaling=False):
        self.num_img = num_img

        self.config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.config.optionxform = str
        self.config.read(args.path_cfg)

        self.config['DEFAULT']['HOME'] = abspath(join(dirname(abspath(__file__)), *[os.pardir]))

        # load configure file
        dir_ld = f"{dir_network}{self.config['DEFAULT']['dir_stdict']}"
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
        file_ld = os.path.join(dir_network, *[cfg_dft['dir_stdict'], f"checkpoints/{cfg_dft['num_cp']}.pth"])
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

            # image scaling
            if (imgs.nelement() / imgs.shape[1] > 1200**2) and self.scaling:
                imgs = TF.resize(imgs, 600, interpolation=InterpolationMode.NEAREST)

            preds = net(imgs)

            om, on = data_test.m, data_test.n
            m, n = imgs.shape[2:4]
            mi = (m - om) // 2
            ni = (n - on) // 2
        return imgs[..., mi:mi + om, ni:ni + on], preds[..., mi:mi + om, ni:ni + on]


if __name__=='__main__':
    _er = EdgeRegion('/home/users/mireiffe/Documents/Python/TeethSeg/cfg/inference.ini', 51)
    img, er = _er.getEr()

    plt.imshow(er > .5, 'gray')
    plt.show()