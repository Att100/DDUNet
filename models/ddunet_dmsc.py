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
 

class DoubleConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = InvertedResidual(in_channels, in_channels, 1, 1)
        self.conv2 = InvertedResidual(in_channels, out_channels, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class DynamicMultiScaleConv2d(nn.Layer):

    def __init__(self, in_channels, out_channels, scale=1, linear_scale=2, use_dw=True):
        super().__init__()

        self.in_conv = BasicConv2d(in_channels, in_channels * scale, 1)

        self.shortcut = BasicConv2d(in_channels*scale, out_channels, 1) if in_channels*scale != out_channels else nn.Identity()

        self.dilated_convs = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels * scale, in_channels * scale, 3, 1, 
                    d, d, in_channels * scale if use_dw else 1, bias_attr=False),
                nn.BatchNorm2D(in_channels * scale),
                nn.ReLU()
            ) for d in [1, 2, 3, 4]
        ])

        self.weights_branch = nn.Sequential(
            nn.Linear(in_channels * scale, in_channels * scale * linear_scale),
            nn.ReLU(),
            nn.Linear(in_channels * scale * linear_scale, 4)
        )

        self.conv = InvertedResidual(in_channels * scale, in_channels * scale, 1, 1)

        self.out_conv = nn.Sequential(
            nn.Conv2D(in_channels * scale, out_channels, 1),
            nn.BatchNorm2D(out_channels)
        )

    def forward(self, x):
        b = x.shape[0]

        x = self.in_conv(x)

        weights_feat = F.adaptive_avg_pool2d(x, 1)
        weights = F.softmax(self.weights_branch(weights_feat.reshape((weights_feat.shape[0], -1))), axis=1)

        out = self.dilated_convs[0](x) *weights[:, 0].reshape((b, 1, 1, 1))
        for i in range(3):
            out += self.dilated_convs[i+1](x) * weights[:, i+1].reshape((b, 1, 1, 1))
        out = self.conv(out)

        out = self.out_conv(out)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out
    

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
    name = 'dmsc'
    
    def __init__(self, in_channels=1, base_channels=8, n_classes=2):
        super().__init__()

        self.out_channels = n_classes if n_classes != 2 else 1 

        # in 
        self.in_conv = BasicConv2d(in_channels, base_channels, 3, 1)

        # encoder
        self.dmsc_1 = DynamicMultiScaleConv2d(base_channels, base_channels)
        self.down_1 = DownSample(base_channels, base_channels*2, mode='conv2d')
        self.dmsc_2 = DynamicMultiScaleConv2d(base_channels*2, base_channels*2)
        self.down_2 = DownSample(base_channels*2, base_channels*4, mode='conv2d')
        self.dmsc_3 = DynamicMultiScaleConv2d(base_channels*4, base_channels*4)
        self.down_3 = DownSample(base_channels*4, base_channels*8, mode='conv2d')
        self.dmsc_4 = DynamicMultiScaleConv2d(base_channels*8, base_channels*8)
        self.down_4 = DownSample(base_channels*8, base_channels*16, mode='conv2d')
        self.dmsc_5 = DynamicMultiScaleConv2d(base_channels*16, base_channels*16)

        # decoder
        self.up_1 = UpSample(base_channels*16, base_channels*8)
        self.double_conv_6 = DoubleConv2d(base_channels*16, base_channels*8)
        self.up_2 = UpSample(base_channels*8, base_channels*4)
        self.double_conv_7 = DoubleConv2d(base_channels*8, base_channels*4)
        self.up_3 = UpSample(base_channels*4, base_channels*2)
        self.double_conv_8 = DoubleConv2d(base_channels*4, base_channels*2)
        self.up_4 = UpSample(base_channels*2, base_channels)
        self.double_conv_9 = DoubleConv2d(base_channels*2, base_channels)

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