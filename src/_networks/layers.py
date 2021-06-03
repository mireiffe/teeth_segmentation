'''
layers
'''
import torch
from torch import nn
from torch.nn.modules.conv import Conv2d


class TestLayer(nn.Module):
    '''
    Conv => BN => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    '''
    [conv2d => BN => ReLU] x 2
    '''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.dbl_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.dbl_conv(x)


class DoubleConvSC(nn.Module):
    '''
    [conv2d => BN => ReLU => conv2D => BN => +x => ReLU]
    '''
    def __init__(self, in_ch, out_ch):
        super(DoubleConvSC, self).__init__()
        self.dbl_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.dbl_conv(x) + x), self.relu(self.dbl_conv(x))


class DownCV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownCV, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.down_conv(x)


class UpLayTC(nn.Module):
    '''
    Upscaling | Short connection => conv2d * 2
    '''
    def __init__(self, in_ch, out_ch, scalefactor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DownDCSC(nn.Module):
    '''
    2x2 Conv with stride 2 => DoubleConvSC
    '''
    def __init__(self, in_ch, out_ch):
        super(DownDCSC, self).__init__()
        self.mp_dcsc = nn.Sequential(
            DownCV(in_ch, out_ch),
            DoubleConvSC(out_ch, out_ch)
        )

    def forward(self, x):
        return self.mp_dcsc(x)

class TransConvDC(nn.Module):
    '''
    Transpose conv => DoubleConv
    '''
    def __init__(self, in_ch, out_ch):
        super(TransConvDC, self).__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dconv = DoubleConv(2 * out_ch, out_ch)

    def forward(self, x_skip, x):
        x = torch.cat((x_skip, self.trans_conv(x)), dim=1)
        return self.dconv(x)


class TransConvResDC(nn.Module):
    '''
    Transpose conv => DoubleConv
    '''
    def __init__(self, in_ch, out_ch):
        super(TransConvResDC, self).__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dconv = DoubleConv(out_ch, out_ch)

    def forward(self, x_skip, x):
        x = x_skip + self.trans_conv(x)
        return self.dconv(x)


class TransConvResDCSC(nn.Module):
    '''
    Transpose conv => DoubleConv
    '''
    def __init__(self, in_ch, out_ch):
        super(TransConvResDCSC, self).__init__()
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dconv = DoubleConvSC(out_ch, out_ch)

    def forward(self, x_skip, x):
        x = x_skip + self.trans_conv(x)
        return self.dconv(x)[0]


class InLayer(nn.Module):
    '''
    Conv => BN => ReLU =>
    Conv => BN => + conv(x, 1x1) => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super(InLayer, self).__init__()
        self.dbl_conv = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.oxo_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.dbl_conv(x) + self.oxo_conv(x)), self.relu(self.dbl_conv(x))


class ConvReLU(nn.Module):
    '''
    ReflectPadding => Conv => BN => ReLU
    '''
    def __init__(self, in_ch, out_ch, ksz, grps):
        super(ConvReLU, self).__init__()
        padsz = (ksz - 1) // 2
        self.conv_relu = nn.Sequential(
            nn.ReflectionPad2d(padding=padsz),
            nn.Conv2d(in_ch, out_ch, kernel_size=ksz, padding=0, bias=False, groups=grps),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_relu(x)


class OutLayer(nn.Module):
    '''
    conv => BN => (activative function)
    '''
    def __init__(self, in_ch, out_ch, act_fun):
        super(OutLayer, self).__init__()
        afun = eval(f'nn.{act_fun}()')
        self.out_lay = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            afun
        )

    def forward(self, x):
        return self.out_lay(x)


class UpLaySC(nn.Module):
    '''
    Upscaling | Short connection => conv2d * 2
    '''
    def __init__(self, in_ch, out_ch, scalefactor=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scalefactor)
        )
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = x_skip + self.up(x)
        x = self.up(x)
        return self.conv(x)
