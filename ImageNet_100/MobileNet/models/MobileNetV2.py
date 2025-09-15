import time
import torch
import torch.nn as nn
from models.DWT.QHWTD import QHWTD

def conv_1x1(inp, oup):
    return nn.Conv2d(inp, oup, 1, 1, 0, bias=False)


def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, ks):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp * expand_ratio,
                    inp * expand_ratio,
                    ks,
                    stride,
                    1,
                    groups=inp * expand_ratio,
                    bias=False,
                ),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_1x1(inp * expand_ratio, oup),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                conv_1x1(inp, inp * expand_ratio),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    inp * expand_ratio,
                    inp * expand_ratio,
                    3,
                    stride,
                    1,
                    groups=inp * expand_ratio,
                    bias=False,
                ),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                conv_1x1(inp * expand_ratio, oup),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        t = x
        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""

    def __init__(self, T, feature_dim, input_size=32, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.interverted_residual_setting = [
            # t, c, n, s,ks
            [1, 16, 1, 1, 3],
            [T, 24, 2, 2, 3],
            [T, 32, 3, 2, 3],
            [T, 64, 4, 2, 3],
            [T, 96, 3, 1, 3],
            [T, 160, 3, 2, 3],
            [T, 320, 1, 1, 3],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Sequential(
            conv_3x3_bn(3, input_channel,2),
            # QHWTD(3,input_channel,3,3,5),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        )

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s, ks in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t, ks)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = nn.Sequential(
            conv_1x1(input_channel, self.last_channel),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.last_channel, feature_dim),
        )
        self._initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        out = self.blocks[2](out)
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
