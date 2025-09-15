import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DWT.QHWTD import QHWTD

wt_levels = [3, 3, 3]


# 定义基础的残差块 (BasicBlock)
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数（ResNet-18/34 为1，ResNet-50/101/152 为4）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        if stride == 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Sequential(
                QHWTD(in_channels, out_channels, 3, wt_levels.pop()),
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 下采样层（用于调整维度）

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整输入维度以匹配输出

        out += identity  # 残差连接
        out = F.relu(out)
        return out


# 定义 ResNet-18
class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)  # -> [B, 64, 56, 56]
        x = self.layer1(x)  # -> [B, 64, 56, 56]
        x = self.layer2(x)  # -> [B, 128, 28, 28]
        x = self.layer3(x)  # -> [B, 256, 14, 14]
        x = self.layer4(x)  # -> [B, 512, 7, 7]

        x = self.avgpool(x)  # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # -> [B, 512]
        x = self.fc(x)  # -> [B, num_classes]
        return x


# 实例化 ResNet-18（输出100类）
def resnet18(num_classes=1000):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    return ResNet18(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# 定义 Bottleneck 残差块（ResNet-50/101/152 使用）
class Bottleneck(nn.Module):
    expansion = 4  # 扩展系数（输出通道数是中间层的4倍）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # # 3x3 卷积（可能下采样）
        if stride == 1:
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
        else:
            self.conv2 = QHWTD(out_channels, out_channels, 3, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 卷积升维
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample  # 下采样层

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整输入维度

        out += identity  # 残差连接
        out = F.relu(out)
        return out


# 定义 ResNet-50
class ResNet50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=100):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # [B, 256, 56, 56]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # [B, 512, 28, 28]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # [B, 1024, 14, 14]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # [B, 2048, 7, 7]

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 判断是否需要下采样（调整维度）
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)  # [B, 64, 56, 56]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # [B, 2048]
        x = self.fc(x)  # [B, num_classes]
        return x


# 实例化 ResNet-50（输出100类）
def resnet50(num_classes=100):
    return ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


# 实例化 ResNet-101
def resnet101(num_classes=100):
    return ResNet50(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


# 测试网络
if __name__ == "__main__":
    if __name__ == "__main__":
        import time

        input = torch.randn(1, 3, 224, 224)
        model = resnet18()
        model.eval()
        # 预热 GPU（避免初次运行时间不准确）
        for _ in range(10):
            with torch.no_grad():
                _ = model(input)

        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            _ = model(input)
        end_time = time.time()

        inference_time = end_time - start_time  # 单帧推理时间（秒）
        fps = 1 / inference_time  # 计算 FPS
        print(f"Inference time: {inference_time:.4f} seconds, FPS: {fps:.2f}")
        from fvcore.nn import FlopCountAnalysis

        x = FlopCountAnalysis(model, input)
        print(x.total())
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 4  # 如果是浮点数就是4
        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para / 1e6) + 'M')
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)
