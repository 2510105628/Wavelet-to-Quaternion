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
        # init.kaiming_normal_(self.r_weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.i_weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.j_weight, mode='fan_out', nonlinearity='relu')
        # init.kaiming_normal_(self.k_weight, mode='fan_out', nonlinearity='relu')
        # if self.scale_param is not None:
        #     torch.nn.init.xavier_uniform_(self.scale_param.data)
        # if self.bias is not None:
        #     self.bias.data.zero_()

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


class _ComplexBatchNorm(Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features  # 每一个部分的channel大小
        self.eps = eps
        self.momentum = momentum
        self.affine = affine  # 是否仿射变换
        self.track_running_stats = track_running_stats  # 是否使用移动均值和移动方差
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(
                    num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    def forward(self, inp):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Crr * n / (n - 1)  #
                        + (1 - exponential_average_factor) * \
                        self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Cii * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                        exponential_average_factor * Cri * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
                      Rrr[None, :, None, None] * inp.real +
                      Rri[None, :, None, None] * inp.imag
              ).type(torch.complex64) + 1j * (
                      Rii[None, :, None, None] * inp.imag +
                      Rri[None, :, None, None] * inp.real
              ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                          self.weight[None, :, 0, None, None] * inp.real
                          + self.weight[None, :, 2, None, None] * inp.imag
                          + self.bias[None, :, 0, None, None]
                  ).type(torch.complex64) + 1j * (
                          self.weight[None, :, 2, None, None] * inp.real
                          + self.weight[None, :, 1, None, None] * inp.imag
                          + self.bias[None, :, 1, None, None]
                  ).type(
                torch.complex64
            )
        return inp


class _QuaternionBatchNorm(Module):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(_QuaternionBatchNorm, self).__init__()
        self.num_features = num_features // 4  # 每一个部分的channel大小
        self.eps = eps
        self.momentum = momentum
        self.affine = affine  # 是否仿射变换
        self.track_running_stats = track_running_stats  # 是否使用移动均值和移动方差
        if self.affine:
            self.weight = Parameter(torch.Tensor(self.num_features, 10))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(
                    1, num_features, 1, 1)
            )
            self.register_buffer("running_covar", torch.zeros(self.num_features, 10))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 0.5
            self.running_covar[:, 1] = 0.5
            self.running_covar[:, 2] = 0.5
            self.running_covar[:, 3] = 0.5
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.zeros_(self.weight)
            init.constant_(self.weight[:, :4], 0.5)
            init.zeros_(self.bias)


class QuaternionBatchNorm(_QuaternionBatchNorm):
    def forward(self, inp):
        ndim = inp.dim()
        inp_shape = inp.size()
        inp_channel = inp_shape[1]
        broadcast_shape = [1] * ndim
        broadcast_shape[1] = inp_channel
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            inp_r = get_r(inp)
            inp_i = get_i(inp)
            inp_j = get_j(inp)
            inp_k = get_k(inp)

            broadcast_mean_shape = [1] * ndim
            broadcast_mean_shape[1] = inp_channel // 4

            mean_r = inp_r.mean([0, 2, 3])
            mean_i = inp_i.mean([0, 2, 3])
            mean_j = inp_j.mean([0, 2, 3])
            mean_k = inp_k.mean([0, 2, 3])
            mean_r = torch.reshape(mean_r, broadcast_mean_shape)
            mean_i = torch.reshape(mean_i, broadcast_mean_shape)
            mean_j = torch.reshape(mean_j, broadcast_mean_shape)
            mean_k = torch.reshape(mean_k, broadcast_mean_shape)
            mean = torch.cat([mean_r, mean_i, mean_j, mean_k], dim=1)

        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )
        broadcast_mean = torch.reshape(mean, broadcast_shape)
        inp = inp - broadcast_mean

        inp_r = get_r(inp)
        inp_i = get_i(inp)
        inp_j = get_j(inp)
        inp_k = get_k(inp)

        if self.training or (not self.track_running_stats):
            # 获取矩阵W
            n = inp_r.numel() / inp_r.size(1)
            Vrr = 1.0 / n * inp_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Vii = 1.0 / n * inp_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Vjj = 1.0 / n * inp_j.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Vkk = 1.0 / n * inp_k.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Vri = (inp_r.mul(inp_i)).mean(dim=[0, 2, 3])
            Vrj = (inp_r.mul(inp_j)).mean(dim=[0, 2, 3])
            Vrk = (inp_r.mul(inp_k)).mean(dim=[0, 2, 3])
            Vij = (inp_i.mul(inp_j)).mean(dim=[0, 2, 3])
            Vik = (inp_i.mul(inp_k)).mean(dim=[0, 2, 3])
            Vjk = (inp_j.mul(inp_k)).mean(dim=[0, 2, 3])

        else:
            Vrr = self.running_covar[:, 0] + self.eps
            Vii = self.running_covar[:, 1] + self.eps
            Vjj = self.running_covar[:, 2] + self.eps
            Vkk = self.running_covar[:, 3] + self.eps
            Vri = self.running_covar[:, 4]
            Vrj = self.running_covar[:, 5]
            Vrk = self.running_covar[:, 6]
            Vij = self.running_covar[:, 7]
            Vik = self.running_covar[:, 8]
            Vjk = self.running_covar[:, 9]
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Vrr * n / (n - 1)  #
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Vii * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 1]
                )
                self.running_covar[:, 2] = (
                        exponential_average_factor * Vjj * n / (n - 1)  #
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 2]
                )

                self.running_covar[:, 3] = (
                        exponential_average_factor * Vkk * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 3]
                )
                self.running_covar[:, 4] = (
                        exponential_average_factor * Vri * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 4]
                )
                self.running_covar[:, 5] = (
                        exponential_average_factor * Vrj * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 5]
                )
                self.running_covar[:, 6] = (
                        exponential_average_factor * Vrk * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 6]
                )
                self.running_covar[:, 7] = (
                        exponential_average_factor * Vij * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 7]
                )
                self.running_covar[:, 8] = (
                        exponential_average_factor * Vik * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 8]
                )
                self.running_covar[:, 9] = (
                        exponential_average_factor * Vjk * n / (n - 1)
                        + (1 - exponential_average_factor) *
                        self.running_covar[:, 9]
                )

        # calculate the inverse square root the covariance matrix
        Wrr = torch.sqrt(Vrr)
        Wri = (1.0 / Wrr) * Vri
        Wii = torch.sqrt((Vii - (Wri * Wri)))
        Wrj = (1.0 / Wrr) * Vrj
        Wij = (1.0 / Wii) * (Vij - (Wri * Wrj))
        Wjj = torch.sqrt((Vjj - (Wij * Wij + Wrj * Wrj)))
        Wrk = (1.0 / Wrr) * Vrk
        Wik = (1.0 / Wii) * (Vik - (Wri * Wrk))
        Wjk = (1.0 / Wjj) * (Vjk - (Wij * Wik + Wrj * Wrk))
        Wkk = torch.sqrt((Vkk - (Wjk * Wjk + Wik * Wik + Wrk * Wrk)))

        broadcast_shape[1] = inp_channel // 4
        broadcast_Wrr = torch.reshape(Wrr, broadcast_shape)
        broadcast_Wri = torch.reshape(Wri, broadcast_shape)
        broadcast_Wii = torch.reshape(Wii, broadcast_shape)
        broadcast_Wrj = torch.reshape(Wrj, broadcast_shape)
        broadcast_Wij = torch.reshape(Wij, broadcast_shape)
        broadcast_Wjj = torch.reshape(Wjj, broadcast_shape)
        broadcast_Wrk = torch.reshape(Wrk, broadcast_shape)
        broadcast_Wik = torch.reshape(Wik, broadcast_shape)
        broadcast_Wjk = torch.reshape(Wjk, broadcast_shape)
        broadcast_Wkk = torch.reshape(Wkk, broadcast_shape)

        cat_W_1 = torch.cat([broadcast_Wrr, broadcast_Wri, broadcast_Wrj, broadcast_Wrk], dim=1)
        cat_W_2 = torch.cat([broadcast_Wri, broadcast_Wii, broadcast_Wij, broadcast_Wik], dim=1)
        cat_W_3 = torch.cat([broadcast_Wrj, broadcast_Wij, broadcast_Wjj, broadcast_Wjk], dim=1)
        cat_W_4 = torch.cat([broadcast_Wrk, broadcast_Wik, broadcast_Wjk, broadcast_Wkk], dim=1)

        input1 = torch.cat([inp_r, inp_r, inp_r, inp_r], dim=1)
        input2 = torch.cat([inp_i, inp_i, inp_i, inp_i], dim=1)
        input3 = torch.cat([inp_j, inp_j, inp_j, inp_j], dim=1)
        input4 = torch.cat([inp_k, inp_k, inp_k, inp_k], dim=1)

        inp = cat_W_1 * input1 + \
              cat_W_2 * input2 + \
              cat_W_3 * input3 + \
              cat_W_4 * input4

        if self.affine:
            broadcast_gamma_rr = torch.reshape(self.weight[:, 0], broadcast_shape)
            broadcast_gamma_ii = torch.reshape(self.weight[:, 1], broadcast_shape)
            broadcast_gamma_jj = torch.reshape(self.weight[:, 2], broadcast_shape)
            broadcast_gamma_kk = torch.reshape(self.weight[:, 3], broadcast_shape)
            broadcast_gamma_ri = torch.reshape(self.weight[:, 4], broadcast_shape)
            broadcast_gamma_rj = torch.reshape(self.weight[:, 5], broadcast_shape)
            broadcast_gamma_rk = torch.reshape(self.weight[:, 6], broadcast_shape)
            broadcast_gamma_ij = torch.reshape(self.weight[:, 7], broadcast_shape)
            broadcast_gamma_ik = torch.reshape(self.weight[:, 8], broadcast_shape)
            broadcast_gamma_jk = torch.reshape(self.weight[:, 9], broadcast_shape)

            cat_gamma_1 = torch.cat([broadcast_gamma_rr,
                                     broadcast_gamma_ri,
                                     broadcast_gamma_rj,
                                     broadcast_gamma_rk], dim=1)
            cat_gamma_2 = torch.cat([broadcast_gamma_ri,
                                     broadcast_gamma_ii,
                                     broadcast_gamma_ij,
                                     broadcast_gamma_ik], dim=1)
            cat_gamma_3 = torch.cat([broadcast_gamma_rj,
                                     broadcast_gamma_ij,
                                     broadcast_gamma_jj,
                                     broadcast_gamma_jk], dim=1)
            cat_gamma_4 = torch.cat([broadcast_gamma_rk,
                                     broadcast_gamma_ik,
                                     broadcast_gamma_jk,
                                     broadcast_gamma_kk], dim=1)

            centred_r = get_r(inp)
            centred_i = get_i(inp)
            centred_j = get_j(inp)
            centred_k = get_k(inp)

            input1 = torch.cat([centred_r, centred_r, centred_r, centred_r], dim=1)
            input2 = torch.cat([centred_i, centred_i, centred_i, centred_i], dim=1)
            input3 = torch.cat([centred_j, centred_j, centred_j, centred_j], dim=1)
            input4 = torch.cat([centred_k, centred_k, centred_k, centred_k], dim=1)

            broadcast_shape[1] = inp_channel
            broadcast_beta = torch.reshape(self.bias, broadcast_shape)

            inp = cat_gamma_1 * input1 + \
                  cat_gamma_2 * input2 + \
                  cat_gamma_3 * input3 + \
                  cat_gamma_4 * input4 + \
                  broadcast_beta

        return inp
