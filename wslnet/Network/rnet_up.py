import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module12(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module12, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module16(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride):
        super(Module16, self).__init__()
        self.conv2d_0_16 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=conv2d_0_stride,
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()

    def construct(self, x, x0):
        opt_conv2d_0 = self.conv2d_0_16(x)
        opt_add_1 = P.Add()(x0, opt_conv2d_0)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module70(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module12_0_conv2d_0_in_channels,
                 module12_0_conv2d_0_out_channels, module12_0_conv2d_0_kernel_size, module12_0_conv2d_0_stride,
                 module12_0_conv2d_0_padding, module12_0_conv2d_0_pad_mode, module12_1_conv2d_0_in_channels,
                 module12_1_conv2d_0_out_channels, module12_1_conv2d_0_kernel_size, module12_1_conv2d_0_stride,
                 module12_1_conv2d_0_padding, module12_1_conv2d_0_pad_mode, module16_0_conv2d_0_in_channels,
                 module16_0_conv2d_0_out_channels, module16_0_conv2d_0_stride, module12_2_conv2d_0_in_channels,
                 module12_2_conv2d_0_out_channels, module12_2_conv2d_0_kernel_size, module12_2_conv2d_0_stride,
                 module12_2_conv2d_0_padding, module12_2_conv2d_0_pad_mode, module12_3_conv2d_0_in_channels,
                 module12_3_conv2d_0_out_channels, module12_3_conv2d_0_kernel_size, module12_3_conv2d_0_stride,
                 module12_3_conv2d_0_padding, module12_3_conv2d_0_pad_mode, module16_1_conv2d_0_in_channels,
                 module16_1_conv2d_0_out_channels, module16_1_conv2d_0_stride):
        super(Module70, self).__init__()
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_0_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_0_conv2d_0_stride,
                                   conv2d_0_padding=module12_0_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_0_conv2d_0_pad_mode)
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_1_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_1_conv2d_0_stride,
                                   conv2d_0_padding=module12_1_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module16_0 = Module16(conv2d_0_in_channels=module16_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module16_0_conv2d_0_out_channels,
                                   conv2d_0_stride=module16_0_conv2d_0_stride)
        self.module12_2 = Module12(conv2d_0_in_channels=module12_2_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_2_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_2_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_2_conv2d_0_stride,
                                   conv2d_0_padding=module12_2_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_2_conv2d_0_pad_mode)
        self.module12_3 = Module12(conv2d_0_in_channels=module12_3_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_3_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_3_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_3_conv2d_0_stride,
                                   conv2d_0_padding=module12_3_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_3_conv2d_0_pad_mode)
        self.module16_1 = Module16(conv2d_0_in_channels=module16_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module16_1_conv2d_0_out_channels,
                                   conv2d_0_stride=module16_1_conv2d_0_stride)

    def construct(self, x):
        module12_0_opt = self.module12_0(x)
        module12_1_opt = self.module12_1(module12_0_opt)
        opt_conv2d_0 = self.conv2d_0(module12_1_opt)
        module16_0_opt = self.module16_0(x, opt_conv2d_0)
        module12_2_opt = self.module12_2(module16_0_opt)
        module12_3_opt = self.module12_3(module12_2_opt)
        module16_1_opt = self.module16_1(module12_3_opt, module16_0_opt)
        return module16_1_opt


class Module19(nn.Cell):
    def __init__(self, module12_0_conv2d_0_in_channels, module12_0_conv2d_0_out_channels,
                 module12_0_conv2d_0_kernel_size, module12_0_conv2d_0_stride, module12_0_conv2d_0_padding,
                 module12_0_conv2d_0_pad_mode, module12_1_conv2d_0_in_channels, module12_1_conv2d_0_out_channels,
                 module12_1_conv2d_0_kernel_size, module12_1_conv2d_0_stride, module12_1_conv2d_0_padding,
                 module12_1_conv2d_0_pad_mode):
        super(Module19, self).__init__()
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_0_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_0_conv2d_0_stride,
                                   conv2d_0_padding=module12_0_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_0_conv2d_0_pad_mode)
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_1_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_1_conv2d_0_stride,
                                   conv2d_0_padding=module12_1_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_1_conv2d_0_pad_mode)

    def construct(self, x):
        module12_0_opt = self.module12_0(x)
        module12_1_opt = self.module12_1(module12_0_opt)
        return module12_1_opt


class Module34(nn.Cell):
    def __init__(self, module16_0_conv2d_0_in_channels, module16_0_conv2d_0_out_channels, module16_0_conv2d_0_stride,
                 module12_0_conv2d_0_in_channels, module12_0_conv2d_0_out_channels, module12_0_conv2d_0_kernel_size,
                 module12_0_conv2d_0_stride, module12_0_conv2d_0_padding, module12_0_conv2d_0_pad_mode,
                 module12_1_conv2d_0_in_channels, module12_1_conv2d_0_out_channels, module12_1_conv2d_0_kernel_size,
                 module12_1_conv2d_0_stride, module12_1_conv2d_0_padding, module12_1_conv2d_0_pad_mode,
                 module16_1_conv2d_0_in_channels, module16_1_conv2d_0_out_channels, module16_1_conv2d_0_stride):
        super(Module34, self).__init__()
        self.module16_0 = Module16(conv2d_0_in_channels=module16_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module16_0_conv2d_0_out_channels,
                                   conv2d_0_stride=module16_0_conv2d_0_stride)
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_0_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_0_conv2d_0_stride,
                                   conv2d_0_padding=module12_0_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_0_conv2d_0_pad_mode)
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_1_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_1_conv2d_0_stride,
                                   conv2d_0_padding=module12_1_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_1_conv2d_0_pad_mode)
        self.module16_1 = Module16(conv2d_0_in_channels=module16_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module16_1_conv2d_0_out_channels,
                                   conv2d_0_stride=module16_1_conv2d_0_stride)

    def construct(self, x, x0):
        module16_0_opt = self.module16_0(x, x0)
        module12_0_opt = self.module12_0(module16_0_opt)
        module12_1_opt = self.module12_1(module12_0_opt)
        module16_1_opt = self.module16_1(module12_1_opt, module16_0_opt)
        return module16_1_opt


class Module39(nn.Cell):
    def __init__(self):
        super(Module39, self).__init__()
        self.relu_2 = nn.ReLU()

    def construct(self, x, x0, x1):
        opt_mul_0 = P.Mul()(x, x0)
        opt_add_1 = P.Add()(opt_mul_0, x1)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module62(nn.Cell):
    def __init__(self):
        super(Module62, self).__init__()
        self.stridedslice_0 = P.StridedSlice()
        self.stridedslice_0_begin = (0, 256, 0, 0)
        self.stridedslice_0_end = (2, 512, 9, 9)
        self.stridedslice_0_strides = (1, 1, 1, 1)
        self.module39_0 = Module39()

    def construct(self, x, x0, x1):
        opt_stridedslice_0 = self.stridedslice_0(x, self.stridedslice_0_begin, self.stridedslice_0_end,
                                                 self.stridedslice_0_strides)
        module39_0_opt = self.module39_0(x0, x1, opt_stridedslice_0)
        return module39_0_opt


class Module3(nn.Cell):
    def __init__(self):
        super(Module3, self).__init__()
        self.concat_0 = P.Concat(axis=1)
        self.conv2d_1 = nn.Conv2d(in_channels=512,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.avgpool2d_3 = nn.AvgPool2d(kernel_size=(9, 9))
        self.pad_maxpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=(9, 9), stride=(9, 9))
        self.squeeze_5 = P.Squeeze(axis=(2, 3))
        self.squeeze_6 = P.Squeeze(axis=(2, 3))
        self.matmul_7_w = Parameter(Tensor(np.random.uniform(0, 1, (256, 64)).astype(np.float32)), name=None)
        self.matmul_8_w = Parameter(Tensor(np.random.uniform(0, 1, (256, 64)).astype(np.float32)), name=None)
        self.batchnorm1d_9 = nn.BatchNorm1d(num_features=64, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.batchnorm1d_10 = nn.BatchNorm1d(num_features=64, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_11 = nn.ReLU()
        self.relu_12 = nn.ReLU()
        self.matmul_13_w = Parameter(Tensor(np.random.uniform(0, 1, (64, 256)).astype(np.float32)), name=None)
        self.matmul_14_w = Parameter(Tensor(np.random.uniform(0, 1, (64, 256)).astype(np.float32)), name=None)
        self.batchnorm1d_15 = nn.BatchNorm1d(num_features=256, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.batchnorm1d_16 = nn.BatchNorm1d(num_features=256, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.sigmoid_18 = nn.Sigmoid()
        self.expanddims_19 = P.ExpandDims()
        self.expanddims_19_axis = -1
        self.expanddims_20 = P.ExpandDims()
        self.expanddims_20_axis = -1

    def construct(self, x, x0):
        opt_concat_0 = self.concat_0((x, x0, ))
        opt_conv2d_1 = self.conv2d_1(opt_concat_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_avgpool2d_3 = self.avgpool2d_3(opt_relu_2)
        opt_maxpool2d_4 = self.pad_maxpool2d_4(opt_relu_2)
        opt_maxpool2d_4 = self.maxpool2d_4(opt_maxpool2d_4)
        opt_squeeze_5 = self.squeeze_5(opt_avgpool2d_3)
        opt_squeeze_6 = self.squeeze_6(opt_maxpool2d_4)
        opt_matmul_7 = P.matmul(opt_squeeze_5, self.matmul_7_w)
        opt_matmul_8 = P.matmul(opt_squeeze_6, self.matmul_8_w)
        opt_batchnorm1d_9 = self.batchnorm1d_9(opt_matmul_7)
        opt_batchnorm1d_10 = self.batchnorm1d_10(opt_matmul_8)
        opt_relu_11 = self.relu_11(opt_batchnorm1d_9)
        opt_relu_12 = self.relu_12(opt_batchnorm1d_10)
        opt_matmul_13 = P.matmul(opt_relu_11, self.matmul_13_w)
        opt_matmul_14 = P.matmul(opt_relu_12, self.matmul_14_w)
        opt_batchnorm1d_15 = self.batchnorm1d_15(opt_matmul_13)
        opt_batchnorm1d_16 = self.batchnorm1d_16(opt_matmul_14)
        opt_add_17 = P.Add()(opt_batchnorm1d_15, opt_batchnorm1d_16)
        opt_sigmoid_18 = self.sigmoid_18(opt_add_17)
        opt_expanddims_19 = self.expanddims_19(opt_sigmoid_18, self.expanddims_19_axis)
        opt_expanddims_20 = self.expanddims_20(opt_expanddims_19, self.expanddims_20_axis)
        opt_mul_21 = P.Mul()(opt_relu_2, opt_expanddims_20)
        opt_add_22 = P.Add()(opt_mul_21, opt_relu_2)
        return opt_add_22


class Module40(nn.Cell):
    def __init__(self, module12_0_conv2d_0_in_channels, module12_0_conv2d_0_out_channels,
                 module12_0_conv2d_0_kernel_size, module12_0_conv2d_0_stride, module12_0_conv2d_0_padding,
                 module12_0_conv2d_0_pad_mode, module12_1_conv2d_0_in_channels, module12_1_conv2d_0_out_channels,
                 module12_1_conv2d_0_kernel_size, module12_1_conv2d_0_stride, module12_1_conv2d_0_padding,
                 module12_1_conv2d_0_pad_mode):
        super(Module40, self).__init__()
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_0_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_0_conv2d_0_stride,
                                   conv2d_0_padding=module12_0_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_0_conv2d_0_pad_mode)
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_1_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_1_conv2d_0_stride,
                                   conv2d_0_padding=module12_1_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module12_0_opt = self.module12_0(x)
        module12_1_opt = self.module12_1(module12_0_opt)
        opt_conv2d_0 = self.conv2d_0(module12_1_opt)
        return opt_conv2d_0


class Module32(nn.Cell):
    def __init__(self):
        super(Module32, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=256,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.reducemean_1 = P.ReduceMean(keep_dims=True)
        self.reducemean_1_axis = 1
        self.concat_2 = P.Concat(axis=1)
        self.concat_2_w = Parameter(Tensor(np.zeros((2, 1, 9, 9)).astype(np.float32)), name=None)
        self.conv2d_3 = nn.Conv2d(in_channels=2,
                                  out_channels=1,
                                  kernel_size=(7, 7),
                                  stride=(1, 1),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.sigmoid_4 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_reducemean_1 = self.reducemean_1(opt_conv2d_0, self.reducemean_1_axis)
        opt_concat_2 = self.concat_2((opt_reducemean_1, self.concat_2_w))
        opt_conv2d_3 = self.conv2d_3(opt_concat_2)
        opt_sigmoid_4 = self.sigmoid_4(opt_conv2d_3)
        opt_mul_5 = P.Mul()(opt_sigmoid_4, opt_conv2d_0)
        return opt_mul_5


class Module69(nn.Cell):
    def __init__(self, module12_0_conv2d_0_in_channels, module12_0_conv2d_0_out_channels,
                 module12_0_conv2d_0_kernel_size, module12_0_conv2d_0_stride, module12_0_conv2d_0_padding,
                 module12_0_conv2d_0_pad_mode, module12_1_conv2d_0_in_channels, module12_1_conv2d_0_out_channels,
                 module12_1_conv2d_0_kernel_size, module12_1_conv2d_0_stride, module12_1_conv2d_0_padding,
                 module12_1_conv2d_0_pad_mode):
        super(Module69, self).__init__()
        self.module12_0 = Module12(conv2d_0_in_channels=module12_0_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_0_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_0_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_0_conv2d_0_stride,
                                   conv2d_0_padding=module12_0_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_0_conv2d_0_pad_mode)
        self.module12_1 = Module12(conv2d_0_in_channels=module12_1_conv2d_0_in_channels,
                                   conv2d_0_out_channels=module12_1_conv2d_0_out_channels,
                                   conv2d_0_kernel_size=module12_1_conv2d_0_kernel_size,
                                   conv2d_0_stride=module12_1_conv2d_0_stride,
                                   conv2d_0_padding=module12_1_conv2d_0_padding,
                                   conv2d_0_pad_mode=module12_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=256,
                                  out_channels=512,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.stridedslice_1 = P.StridedSlice()
        self.stridedslice_1_begin = (0, 0, 0, 0)
        self.stridedslice_1_end = (2, 256, 9, 9)
        self.stridedslice_1_strides = (1, 1, 1, 1)
        self.stridedslice_2 = P.StridedSlice()
        self.stridedslice_2_begin = (0, 256, 0, 0)
        self.stridedslice_2_end = (2, 512, 9, 9)
        self.stridedslice_2_strides = (1, 1, 1, 1)
        self.module39_0 = Module39()

    def construct(self, x):
        module12_0_opt = self.module12_0(x)
        module12_1_opt = self.module12_1(module12_0_opt)
        opt_conv2d_0 = self.conv2d_0(module12_1_opt)
        opt_stridedslice_1 = self.stridedslice_1(opt_conv2d_0, self.stridedslice_1_begin, self.stridedslice_1_end,
                                                 self.stridedslice_1_strides)
        opt_stridedslice_2 = self.stridedslice_2(opt_conv2d_0, self.stridedslice_2_begin, self.stridedslice_2_end,
                                                 self.stridedslice_2_strides)
        module39_0_opt = self.module39_0(opt_stridedslice_1, module12_1_opt, opt_stridedslice_2)
        return module39_0_opt


class WSLnet_up(nn.Cell):
    def __init__(self):
        super(WSLnet_up, self).__init__()
        self.concat_0 = P.Concat(axis=1)
        self.module12_0 = Module12(conv2d_0_in_channels=4,
                                   conv2d_0_out_channels=64,
                                   conv2d_0_kernel_size=(7, 7),
                                   conv2d_0_stride=(2, 2),
                                   conv2d_0_padding=(3, 3, 3, 3),
                                   conv2d_0_pad_mode="pad")
        self.pad_maxpool2d_3 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module70_0 = Module70(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_in_channels=64,
                                   module12_0_conv2d_0_out_channels=64,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=64,
                                   module12_1_conv2d_0_out_channels=64,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_0_conv2d_0_in_channels=64,
                                   module16_0_conv2d_0_out_channels=256,
                                   module16_0_conv2d_0_stride=(1, 1),
                                   module12_2_conv2d_0_in_channels=256,
                                   module12_2_conv2d_0_out_channels=64,
                                   module12_2_conv2d_0_kernel_size=(1, 1),
                                   module12_2_conv2d_0_stride=(1, 1),
                                   module12_2_conv2d_0_padding=0,
                                   module12_2_conv2d_0_pad_mode="valid",
                                   module12_3_conv2d_0_in_channels=64,
                                   module12_3_conv2d_0_out_channels=64,
                                   module12_3_conv2d_0_kernel_size=(3, 3),
                                   module12_3_conv2d_0_stride=(1, 1),
                                   module12_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_3_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=64,
                                   module16_1_conv2d_0_out_channels=256,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module19_0 = Module19(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=64,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=64,
                                   module12_1_conv2d_0_out_channels=64,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module16_0 = Module16(conv2d_0_in_channels=64, conv2d_0_out_channels=256, conv2d_0_stride=(1, 1))
        self.module70_1 = Module70(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(2, 2),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_0_conv2d_0_in_channels=256,
                                   module16_0_conv2d_0_out_channels=512,
                                   module16_0_conv2d_0_stride=(2, 2),
                                   module12_2_conv2d_0_in_channels=512,
                                   module12_2_conv2d_0_out_channels=128,
                                   module12_2_conv2d_0_kernel_size=(1, 1),
                                   module12_2_conv2d_0_stride=(1, 1),
                                   module12_2_conv2d_0_padding=0,
                                   module12_2_conv2d_0_pad_mode="valid",
                                   module12_3_conv2d_0_in_channels=128,
                                   module12_3_conv2d_0_out_channels=128,
                                   module12_3_conv2d_0_kernel_size=(3, 3),
                                   module12_3_conv2d_0_stride=(1, 1),
                                   module12_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_3_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=128,
                                   module16_1_conv2d_0_out_channels=512,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module19_1 = Module19(module12_0_conv2d_0_in_channels=512,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module34_0 = Module34(module16_0_conv2d_0_in_channels=128,
                                   module16_0_conv2d_0_out_channels=512,
                                   module16_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_in_channels=512,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=128,
                                   module16_1_conv2d_0_out_channels=512,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module70_2 = Module70(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   module12_0_conv2d_0_in_channels=512,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(2, 2),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_0_conv2d_0_in_channels=512,
                                   module16_0_conv2d_0_out_channels=1024,
                                   module16_0_conv2d_0_stride=(2, 2),
                                   module12_2_conv2d_0_in_channels=1024,
                                   module12_2_conv2d_0_out_channels=256,
                                   module12_2_conv2d_0_kernel_size=(1, 1),
                                   module12_2_conv2d_0_stride=(1, 1),
                                   module12_2_conv2d_0_padding=0,
                                   module12_2_conv2d_0_pad_mode="valid",
                                   module12_3_conv2d_0_in_channels=256,
                                   module12_3_conv2d_0_out_channels=256,
                                   module12_3_conv2d_0_kernel_size=(3, 3),
                                   module12_3_conv2d_0_stride=(1, 1),
                                   module12_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_3_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=256,
                                   module16_1_conv2d_0_out_channels=1024,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module19_2 = Module19(module12_0_conv2d_0_in_channels=1024,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module34_1 = Module34(module16_0_conv2d_0_in_channels=256,
                                   module16_0_conv2d_0_out_channels=1024,
                                   module16_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_in_channels=1024,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=256,
                                   module16_1_conv2d_0_out_channels=1024,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module19_3 = Module19(module12_0_conv2d_0_in_channels=1024,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module34_2 = Module34(module16_0_conv2d_0_in_channels=256,
                                   module16_0_conv2d_0_out_channels=1024,
                                   module16_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_in_channels=1024,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=256,
                                   module16_1_conv2d_0_out_channels=1024,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module70_3 = Module70(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   module12_0_conv2d_0_in_channels=1024,
                                   module12_0_conv2d_0_out_channels=512,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=512,
                                   module12_1_conv2d_0_out_channels=512,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(2, 2),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad",
                                   module16_0_conv2d_0_in_channels=1024,
                                   module16_0_conv2d_0_out_channels=2048,
                                   module16_0_conv2d_0_stride=(2, 2),
                                   module12_2_conv2d_0_in_channels=2048,
                                   module12_2_conv2d_0_out_channels=512,
                                   module12_2_conv2d_0_kernel_size=(1, 1),
                                   module12_2_conv2d_0_stride=(1, 1),
                                   module12_2_conv2d_0_padding=0,
                                   module12_2_conv2d_0_pad_mode="valid",
                                   module12_3_conv2d_0_in_channels=512,
                                   module12_3_conv2d_0_out_channels=512,
                                   module12_3_conv2d_0_kernel_size=(3, 3),
                                   module12_3_conv2d_0_stride=(1, 1),
                                   module12_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_3_conv2d_0_pad_mode="pad",
                                   module16_1_conv2d_0_in_channels=512,
                                   module16_1_conv2d_0_out_channels=2048,
                                   module16_1_conv2d_0_stride=(1, 1))
        self.module19_4 = Module19(module12_0_conv2d_0_in_channels=2048,
                                   module12_0_conv2d_0_out_channels=512,
                                   module12_0_conv2d_0_kernel_size=(1, 1),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=0,
                                   module12_0_conv2d_0_pad_mode="valid",
                                   module12_1_conv2d_0_in_channels=512,
                                   module12_1_conv2d_0_out_channels=512,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module16_1 = Module16(conv2d_0_in_channels=512, conv2d_0_out_channels=2048, conv2d_0_stride=(1, 1))
        self.concat_120 = P.Concat(axis=1)
        self.module12_1 = Module12(conv2d_0_in_channels=2048,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.conv2d_122 = nn.Conv2d(in_channels=2048,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_128 = P.StridedSlice()
        self.stridedslice_128_begin = (0, 0, 0, 0)
        self.stridedslice_128_end = (2, 256, 9, 9)
        self.stridedslice_128_strides = (1, 1, 1, 1)
        self.module62_0 = Module62()
        self.module12_2 = Module12(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid")
        self.reducemean_123 = P.ReduceMean(keep_dims=True)
        self.reducemean_123_axis = (2, 3)
        self.module12_3 = Module12(conv2d_0_in_channels=2048,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid")
        self.conv2d_140 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.sigmoid_148 = nn.Sigmoid()
        self.module12_4 = Module12(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.conv2d_199 = nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_206 = P.StridedSlice()
        self.stridedslice_206_begin = (0, 0, 0, 0)
        self.stridedslice_206_end = (2, 256, 9, 9)
        self.stridedslice_206_strides = (1, 1, 1, 1)
        self.module62_1 = Module62()
        self.module12_5 = Module12(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.module12_6 = Module12(conv2d_0_in_channels=4096,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.resizebilinear_136 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.module3_0 = Module3()
        self.module40_0 = Module40(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.sigmoid_261 = nn.Sigmoid()
        self.sub_265_bias = 1.0
        self.conv2d_239 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_276 = nn.ReLU()
        self.module32_0 = Module32()
        self.module69_0 = Module69(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module12_7 = Module12(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.module12_8 = Module12(conv2d_0_in_channels=4096,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.resizebilinear_137 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.module3_1 = Module3()
        self.module40_1 = Module40(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.sigmoid_262 = nn.Sigmoid()
        self.sub_267_bias = 1.0
        self.conv2d_295 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_277 = nn.ReLU()
        self.module32_1 = Module32()
        self.module69_1 = Module69(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.module12_9 = Module12(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=256,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_stride=(1, 1),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad")
        self.module12_10 = Module12(conv2d_0_in_channels=4096,
                                    conv2d_0_out_channels=256,
                                    conv2d_0_kernel_size=(3, 3),
                                    conv2d_0_stride=(1, 1),
                                    conv2d_0_padding=(1, 1, 1, 1),
                                    conv2d_0_pad_mode="pad")
        self.resizebilinear_138 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.module3_2 = Module3()
        self.module40_2 = Module40(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=128,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=128,
                                   module12_1_conv2d_0_out_channels=128,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.sigmoid_263 = nn.Sigmoid()
        self.sub_269_bias = 1.0
        self.conv2d_312 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_278 = nn.ReLU()
        self.module32_2 = Module32()
        self.module69_2 = Module69(module12_0_conv2d_0_in_channels=256,
                                   module12_0_conv2d_0_out_channels=256,
                                   module12_0_conv2d_0_kernel_size=(3, 3),
                                   module12_0_conv2d_0_stride=(1, 1),
                                   module12_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_0_conv2d_0_pad_mode="pad",
                                   module12_1_conv2d_0_in_channels=256,
                                   module12_1_conv2d_0_out_channels=256,
                                   module12_1_conv2d_0_kernel_size=(3, 3),
                                   module12_1_conv2d_0_stride=(1, 1),
                                   module12_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module12_1_conv2d_0_pad_mode="pad")
        self.conv2d_230 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_235 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_292 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_294 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_309 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_311 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_325 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_326 = P.ResizeBilinear(size=(288, 288), align_corners=False)

    def construct(self, auto_legalized__0, auto_legalized__1, x1):
        opt_concat_0 = self.concat_0((auto_legalized__0, auto_legalized__1, ))
        module12_0_opt = self.module12_0(opt_concat_0)
        opt_maxpool2d_3 = self.pad_maxpool2d_3(module12_0_opt)
        opt_maxpool2d_3 = self.maxpool2d_3(opt_maxpool2d_3)
        module70_0_opt = self.module70_0(opt_maxpool2d_3)
        module19_0_opt = self.module19_0(module70_0_opt)
        module16_0_opt = self.module16_0(module19_0_opt, module70_0_opt)
        module70_1_opt = self.module70_1(module16_0_opt)
        module19_1_opt = self.module19_1(module70_1_opt)
        module34_0_opt = self.module34_0(module19_1_opt, module70_1_opt)
        module70_2_opt = self.module70_2(module34_0_opt)
        module19_2_opt = self.module19_2(module70_2_opt)
        module34_1_opt = self.module34_1(module19_2_opt, module70_2_opt)
        module19_3_opt = self.module19_3(module34_1_opt)
        module34_2_opt = self.module34_2(module19_3_opt, module34_1_opt)
        module70_3_opt = self.module70_3(module34_2_opt)
        module19_4_opt = self.module19_4(module70_3_opt)
        module16_1_opt = self.module16_1(module19_4_opt, module70_3_opt)
        opt_concat_120 = self.concat_120((module16_1_opt, x1, ))
        module12_1_opt = self.module12_1(module16_1_opt)
        opt_conv2d_122 = self.conv2d_122(module16_1_opt)
        opt_stridedslice_128 = self.stridedslice_128(opt_conv2d_122, self.stridedslice_128_begin,
                                                     self.stridedslice_128_end, self.stridedslice_128_strides)
        module62_0_opt = self.module62_0(opt_conv2d_122, opt_stridedslice_128, module12_1_opt)
        module12_2_opt = self.module12_2(module62_0_opt)
        opt_reducemean_123 = self.reducemean_123(module16_1_opt, self.reducemean_123_axis)
        module12_3_opt = self.module12_3(opt_reducemean_123)
        opt_conv2d_140 = self.conv2d_140(module12_3_opt)
        opt_sigmoid_148 = self.sigmoid_148(opt_conv2d_140)
        opt_mul_172 = P.Mul()(module12_2_opt, opt_sigmoid_148)
        module12_4_opt = self.module12_4(opt_mul_172)
        opt_conv2d_199 = self.conv2d_199(module12_4_opt)
        opt_stridedslice_206 = self.stridedslice_206(opt_conv2d_199, self.stridedslice_206_begin,
                                                     self.stridedslice_206_end, self.stridedslice_206_strides)
        module62_1_opt = self.module62_1(opt_conv2d_199, opt_stridedslice_206, module12_4_opt)
        module12_5_opt = self.module12_5(module62_1_opt)
        module12_6_opt = self.module12_6(opt_concat_120)
        opt_resizebilinear_136 = self.resizebilinear_136(module12_6_opt)
        module3_0_opt = self.module3_0(module12_6_opt, opt_resizebilinear_136)
        module40_0_opt = self.module40_0(module3_0_opt)
        opt_sigmoid_261 = self.sigmoid_261(module40_0_opt)
        opt_mul_264 = P.Mul()(module12_6_opt, opt_sigmoid_261)
        opt_sub_265 = self.sub_265_bias - opt_sigmoid_261
        opt_mul_270 = P.Mul()(opt_resizebilinear_136, opt_sub_265)
        opt_add_273 = P.Add()(opt_mul_264, opt_mul_270)
        opt_conv2d_239 = self.conv2d_239(module12_5_opt)
        opt_relu_276 = self.relu_276(opt_add_273)
        opt_mul_279 = P.Mul()(opt_conv2d_239, opt_relu_276)
        module32_0_opt = self.module32_0(opt_resizebilinear_136)
        opt_add_280 = P.Add()(module32_0_opt, opt_mul_279)
        module69_0_opt = self.module69_0(opt_add_280)
        module12_7_opt = self.module12_7(module69_0_opt)
        module12_8_opt = self.module12_8(opt_concat_120)
        opt_resizebilinear_137 = self.resizebilinear_137(module12_8_opt)
        module3_1_opt = self.module3_1(module12_8_opt, opt_resizebilinear_137)
        module40_1_opt = self.module40_1(module3_1_opt)
        opt_sigmoid_262 = self.sigmoid_262(module40_1_opt)
        opt_mul_266 = P.Mul()(module12_8_opt, opt_sigmoid_262)
        opt_sub_267 = self.sub_267_bias - opt_sigmoid_262
        opt_mul_271 = P.Mul()(opt_resizebilinear_137, opt_sub_267)
        opt_add_274 = P.Add()(opt_mul_266, opt_mul_271)
        opt_conv2d_295 = self.conv2d_295(module12_7_opt)
        opt_relu_277 = self.relu_277(opt_add_274)
        opt_mul_296 = P.Mul()(opt_conv2d_295, opt_relu_277)
        module32_1_opt = self.module32_1(opt_resizebilinear_137)
        opt_add_297 = P.Add()(module32_1_opt, opt_mul_296)
        module69_1_opt = self.module69_1(opt_add_297)
        module12_9_opt = self.module12_9(module69_1_opt)
        module12_10_opt = self.module12_10(opt_concat_120)
        opt_resizebilinear_138 = self.resizebilinear_138(module12_10_opt)
        module3_2_opt = self.module3_2(module12_10_opt, opt_resizebilinear_138)
        module40_2_opt = self.module40_2(module3_2_opt)
        opt_sigmoid_263 = self.sigmoid_263(module40_2_opt)
        opt_mul_268 = P.Mul()(module12_10_opt, opt_sigmoid_263)
        opt_sub_269 = self.sub_269_bias - opt_sigmoid_263
        opt_mul_272 = P.Mul()(opt_resizebilinear_138, opt_sub_269)
        opt_add_275 = P.Add()(opt_mul_268, opt_mul_272)
        opt_conv2d_312 = self.conv2d_312(module12_9_opt)
        opt_relu_278 = self.relu_278(opt_add_275)
        opt_mul_313 = P.Mul()(opt_conv2d_312, opt_relu_278)
        module32_2_opt = self.module32_2(opt_resizebilinear_138)
        opt_add_314 = P.Add()(module32_2_opt, opt_mul_313)
        module69_2_opt = self.module69_2(opt_add_314)
        opt_conv2d_230 = self.conv2d_230(module62_1_opt)
        opt_resizebilinear_235 = self.resizebilinear_235(opt_conv2d_230)
        opt_conv2d_292 = self.conv2d_292(module69_0_opt)
        opt_resizebilinear_294 = self.resizebilinear_294(opt_conv2d_292)
        opt_conv2d_309 = self.conv2d_309(module69_1_opt)
        opt_resizebilinear_311 = self.resizebilinear_311(opt_conv2d_309)
        opt_conv2d_325 = self.conv2d_325(module69_2_opt)
        opt_resizebilinear_326 = self.resizebilinear_326(opt_conv2d_325)
        return opt_resizebilinear_235, opt_resizebilinear_294, opt_resizebilinear_311, opt_resizebilinear_326
