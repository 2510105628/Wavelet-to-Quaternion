import torch
import torch.nn as nn
from models.DWT.QHWTD import QHWTD


def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()  # 转置并合并
    x = x.view(batch, -1, height, width)
    return x


class ShuffleBlock(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.branch1 = nn.Sequential(
            # 1x1卷积降维
            nn.Conv2d(inp, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True),

            # 3x3深度可分离卷积
            nn.Conv2d(oup // 2, oup // 2, 3, 1, 1, groups=oup // 2, bias=False),
            nn.BatchNorm2d(oup // 2),

            # 1x1卷积升维
            nn.Conv2d(oup // 2, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            # 右分支直接通道缩减
            nn.Conv2d(inp, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)  # 按输出通道数分组


class ShuffleBlockStride2(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        # 左分支（3x3卷积下采样）
        self.branch1 = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 2, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True)
        )

        # 右分支（1x1卷积下采样）
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // 2, oup // 2, 3, 2, 1, groups=oup // 2, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.Conv2d(oup // 2, oup // 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=100, scale=1.0):
        super().__init__()
        channels = {
            0.5: [24, 48, 96, 192, 1024],
            1.0: [24, 116, 232, 464, 1024],
            1.5: [24, 176, 352, 704, 1024],
            2.0: [24, 244, 488, 976, 2048]
        }[scale]

        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            QHWTD(3, channels[0], 3, 3, 5),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.Sequential(
            self._make_stage(channels[0], channels[1], 3),
            self._make_stage(channels[1], channels[2], 7),
            self._make_stage(channels[2], channels[3], 3),
            self._make_stage(channels[3], channels[4], 1, last=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[4], num_classes)
        )

    def _make_stage(self, inp, oup, repeats, last=False):
        blocks = []
        blocks.append(ShuffleBlockStride2(inp, oup))
        for _ in range(repeats - 1):
            blocks.append(ShuffleBlock(oup, oup))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)  # [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.classifier(x)
        return x
