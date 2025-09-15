import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from .util import wavelet
from ..QCNN.quaternion_layers import QuaternionConv

class QHWTD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, wt_kernel_size=3, wt_levels=2, wt_type='db1'):
        super(QHWTD, self).__init__()

        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels,
                                                                        torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wavelet_convs = nn.ModuleList(
            # [nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, groups=1,
            #            bias=False) for _ in range(self.wt_levels)]
            [QuaternionConv(in_channels * 4, in_channels * 4, wt_kernel_size, 1, (wt_kernel_size - 1) // 2, bias=False)
             for _ in
             range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            # [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
            [nn.BatchNorm2d(in_channels*4) for _ in range(self.wt_levels)]
        )
        self.downSample = nn.Sequential(
            nn.Conv2d(in_channels+in_channels, out_channels, kernel_size, 2, (kernel_size - 1) // 2, bias=False),
        )

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            #
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        #
        next_x_ll = 0
        #
        x_wt = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            if i > 0:  # 如果不是最浅层（第一级），正常进行逆变换
                curr_x_ll = curr_x_ll + next_x_ll
                curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
                next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)
                next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            else:  # 如果是第一级，保留四个子带
                curr_x_ll = curr_x_ll + next_x_ll
                curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
                x_wt = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)
                x_wt = x_wt[:, :, :curr_shape[2], :curr_shape[3]]
        assert len(x_ll_in_levels) == 0
        x = torch.cat([x, x_wt], dim=1)
        x = self.downSample(x)
        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
