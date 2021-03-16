'''
Model structures
'''
from torch import nn, hub
from torch.nn import functional as F
from torchvision import models

from _networks import layers

from _networks import resnest


class ForTest(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.tl = layers.TestLayer(in_ch, out_ch)
    
    def forward(self, x):
        x = self.tl(x)
        return x

class resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet50 = models.resnet50(pretrained=True)

        self.resnet50

        _ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(_ftrs, 5)

    def forward(self, x):
        self.resnet50(x)


class ResUNet5(nn.Module):
    # Network structure
    CFG_PCH = [64, 128, 256, 512, 1024, 2048, 2048,
              1024, 512, 256, 128, 64, 0]
    CFG_TYPE = ['in', 'down', 'down', 'down', 'down', 'downcv', 'dc',
                'up', 'up', 'up', 'up', 'up', 'out']
    SKIP_LAYERS = [[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]]

    def __init__(self, in_ch, out_ch, act_fun='Sigmoid'):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.act_fun = act_fun

        CFG_CH = [out_ch if x == 0 else x for x in self.CFG_PCH]
        layers = self.makelayers(self.CFG_TYPE, CFG_CH, in_ch, act_fun)
        self.layerlst = nn.ModuleList(layers)
        self.sk_lay = self.SKIP_LAYERS

    def forward(self, x):
        skip_x = []
        for i_lay, f in enumerate(self.layerlst):
            if i_lay in self.sk_lay[0]:
                x, x_s = f(x)
                skip_x += [x_s]
            elif i_lay in self.sk_lay[1]:
                x = f(skip_x.pop(), x)
            else:
                x = f(x)
        return x

    @staticmethod
    def makelayers(cfg_t, cfg_c, in_ch, act_fun):
        '''
        making layers according to the CFG lists
        '''
        lays = []
        for tp_lay, curr_ch in zip(cfg_t, cfg_c):
            if tp_lay == 'in':
                lays += [layers.InLayer(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'down':
                lays += [layers.DownDCSC(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'dcsc':
                lays += [layers.DoubleConvSC(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'downcv':
                lays += [layers.DownCV(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'dc':
                lays += [layers.DoubleConv(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'up':
                lays += [layers.TransConvDC(in_ch, curr_ch)]
                in_ch = curr_ch
            elif tp_lay == 'out':
                lays += [layers.OutLayer(in_ch, curr_ch, act_fun)]
                in_ch = curr_ch
        return lays


class ResNeSt200(nn.Module):
    def __init__(self, in_ch, out_ch, act_fun='Sigmoid'):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.act_fun = act_fun

        self._rsnst = resnest.resnest200(pretrained=False)

        self.up1 = layers.UpLaySC(2048, 1024)
        self.up2 = layers.UpLaySC(1024, 512)
        self.up3 = layers.UpLaySC(512, 256)
        self.up4 = layers.UpLaySC(256, 128)
        self.up5 = layers.UpLaySC(128, 64)      # 256 x 256

        self.out = layers.OutLayer(64, 1, act_fun)


        # hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # resnest = hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=False)

    def forward(self, x):
        m, n = x.shape[2:]
        x  = self._rsnst(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        if m != x.shape[2] or n != x.shape[3]:
            x = F.upsample_bilinear(x, (m, n))
        
        return self.out(x)
