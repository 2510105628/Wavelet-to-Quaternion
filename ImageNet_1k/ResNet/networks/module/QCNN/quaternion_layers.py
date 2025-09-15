##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module, init
from .quaternion_ops import *
from typing import Optional
from numpy.random import RandomState


class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilatation=1, bias=False, init_criterion='he',
                 weight_init='quaternion', operation='convolution2d', quaternion_format=True, ):
        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.rng = RandomState(3407)
        print()
        self.operation = operation
        self.quaternion_format = quaternion_format
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(self.operation,
                                                                       self.in_channels, self.out_channels, kernel_size)

        self.r_weight = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = Parameter(torch.Tensor(*self.w_shape))
        self.register_buffer('running_kernel', torch.zeros(out_channels, in_channels, 1, 1))
        self.cat_kernels_4_r = torch.cat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=1)
        self.cat_kernels_4_i = torch.cat([self.i_weight, self.r_weight, -self.k_weight, self.j_weight], dim=1)
        self.cat_kernels_4_j = torch.cat([self.j_weight, self.k_weight, self.r_weight, -self.i_weight], dim=1)
        self.cat_kernels_4_k = torch.cat([self.k_weight, -self.j_weight, self.i_weight, self.r_weight], dim=1)
        self.cat_kernels_4_quaternion = torch.cat([self.cat_kernels_4_r, self.cat_kernels_4_i, self.cat_kernels_4_j,
                                                   self.cat_kernels_4_k], dim=0)
        self.running_kernel = self.cat_kernels_4_quaternion
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                         self.kernel_size, self.rng, self.init_criterion)

    def quaternion_conv(self, input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                        padding, groups):
        """
        Applies a quaternion convolution to the incoming data:
        """
        cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
        cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
        cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
        cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k],
                                             dim=0)
        self.running_kernel = cat_kernels_4_quaternion
        return F.conv2d(input, cat_kernels_4_quaternion, bias, stride, padding, groups=groups)

    def forward(self, input):
        if self.training:
            return self.quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                                                self.k_weight, self.bias, self.stride, self.padding, self.groups)
        else:
            return F.conv2d(input, self.running_kernel, self.bias, self.stride, self.padding, self.groups)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', bias=' + str(self.bias is not None) \
            + ', kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', q_format=' + str(self.quaternion_format) \
            + ', operation=' + str(self.operation) + ')'




