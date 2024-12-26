import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Iterable

from paddle.vision.models.mobilenetv2 import InvertedResidual

class BasicConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, groups=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        return F.relu(out)
    

class DownSample(nn.Layer):
    def __init__(self, in_channels, out_channels, mode='pooling'):
        super().__init__()
        if mode == 'pooling':
            self.downsample = nn.Sequential(
                nn.MaxPool2D(3, 2, 1),
                BasicConv2d(in_channels, out_channels, 3, 1)
            )
        else:
            self.downsample = BasicConv2d(in_channels, out_channels, 3, 1, 2)

    def forward(self, x):
        return self.downsample(x)


class UpSample(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out


class DDUNet(nn.Layer):
    name = 'baseline'
    
    def __init__(self, in_channels=1, base_channels=8, n_classes=2):
        super().__init__()

        self.out_channels = n_classes if n_classes != 2 else 1 

        # in 
        self.in_conv = BasicConv2d(in_channels, base_channels, 3, 1)

        # encoder
        self.dmsc_1 = BasicConv2d(base_channels, base_channels, 3, 1, 1)
        self.down_1 = DownSample(base_channels, base_channels*2, mode='conv2d')
        self.dmsc_2 = BasicConv2d(base_channels*2, base_channels*2, 3, 1, 1)
        self.down_2 = DownSample(base_channels*2, base_channels*4, mode='conv2d')
        self.dmsc_3 = BasicConv2d(base_channels*4, base_channels*4, 3, 1, 1)
        self.down_3 = DownSample(base_channels*4, base_channels*8, mode='conv2d')
        self.dmsc_4 = BasicConv2d(base_channels*8, base_channels*8, 3, 1, 1)
        self.down_4 = DownSample(base_channels*8, base_channels*16, mode='conv2d')
        self.dmsc_5 = BasicConv2d(base_channels*16, base_channels*16, 3, 1, 1)

        # decoder
        self.up_1 = UpSample(base_channels*16, base_channels*8)
        self.double_conv_6 = InvertedResidual(base_channels*16, base_channels*8, 1, 1)
        self.up_2 = UpSample(base_channels*8, base_channels*4)
        self.double_conv_7 = InvertedResidual(base_channels*8, base_channels*4, 1, 1)
        self.up_3 = UpSample(base_channels*4, base_channels*2)
        self.double_conv_8 = InvertedResidual(base_channels*4, base_channels*2, 1, 1)
        self.up_4 = UpSample(base_channels*2, base_channels)
        self.double_conv_9 = InvertedResidual(base_channels*2, base_channels, 1, 1)

        # classifier
        self.classifier1 = nn.Conv2D(base_channels, self.out_channels, 3, 1, 1)

    def forward(self, x):
        out = self.in_conv(x)
        feat_1 = self.dmsc_1(out)
        out = self.down_1(feat_1)
        feat_2 = self.dmsc_2(out)
        out = self.down_2(feat_2)
        feat_3 = self.dmsc_3(out)
        out = self.down_3(feat_3)
        feat_4 = self.dmsc_4(out)
        out = self.down_4(feat_4)

        feat_last = self.dmsc_5(out)

        out = self.up_1(feat_last)
        out = paddle.concat([out, feat_4], axis=1)
        out = self.double_conv_6(out)
        out = self.up_2(out)
        out = paddle.concat([out, feat_3], axis=1)
        up_x4 = self.double_conv_7(out)
        out = self.up_3(up_x4)
        out = paddle.concat([out, feat_2], axis=1)
        up_x2 = self.double_conv_8(out)
        out = self.up_4(up_x2)
        out = paddle.concat([out, feat_1], axis=1)
        out = self.double_conv_9(out)

        out = self.classifier1(out)
        return out