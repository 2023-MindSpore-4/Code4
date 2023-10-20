import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module9(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module9, self).__init__()
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


class Module4(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
        super(Module4, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(opt_conv2d_4, x)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module17(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module17, self).__init__()
        self.pad_avgpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        opt_avgpool2d_0 = self.pad_avgpool2d_0(x)
        opt_avgpool2d_0 = self.avgpool2d_0(opt_avgpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_avgpool2d_0)
        return opt_conv2d_1


class Module15(nn.Cell):
    def __init__(self, resizebilinear_0_size, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module15, self).__init__()
        self.resizenearestneighbor_0 = P.ResizeNearestNeighbor(size=resizebilinear_0_size, align_corners=False)
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        opt_resizenearestneighbor_0 = self.resizenearestneighbor_0(x)
        opt_conv2d_1 = self.conv2d_1(opt_resizenearestneighbor_0)
        return opt_conv2d_1


class Module12(nn.Cell):
    def __init__(self, batchnorm2d_0_num_features):
        super(Module12, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=9.999999747378752e-06,
                                            momentum=0.8999999761581421)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class Snet(nn.Cell):
    def __init__(self):
        super(Snet, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module9_0 = Module9(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module9_1 = Module9(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_19 = nn.Conv2d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_7 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_21 = nn.ReLU()
        self.module4_0 = Module4(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256)
        self.module4_1 = Module4(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256)
        self.module9_2 = Module9(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module9_3 = Module9(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_61 = nn.Conv2d(in_channels=128,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_37 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_69 = nn.ReLU()
        self.module4_2 = Module4(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512)
        self.module4_3 = Module4(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512)
        self.module4_4 = Module4(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512)
        self.module9_4 = Module9(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module9_5 = Module9(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_129 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_105 = nn.Conv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_137 = nn.ReLU()
        self.module4_5 = Module4(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module4_6 = Module4(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module4_7 = Module4(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module4_8 = Module4(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module4_9 = Module4(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024)
        self.module9_6 = Module9(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.module9_7 = Module9(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=512,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_212 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_188 = nn.Conv2d(in_channels=1024,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_220 = nn.ReLU()
        self.module4_10 = Module4(conv2d_0_in_channels=2048,
                                  conv2d_0_out_channels=512,
                                  conv2d_2_in_channels=512,
                                  conv2d_2_out_channels=512,
                                  conv2d_4_in_channels=512,
                                  conv2d_4_out_channels=2048)
        self.module4_11 = Module4(conv2d_0_in_channels=2048,
                                  conv2d_0_out_channels=512,
                                  conv2d_2_in_channels=512,
                                  conv2d_2_out_channels=512,
                                  conv2d_4_in_channels=512,
                                  conv2d_4_out_channels=2048)
        self.module9_8 = Module9(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module9_9 = Module9(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_11 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module17_0 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=64)
        self.conv2d_47 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module15_0 = Module15(resizebilinear_0_size=(160, 160),
                                   conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=64)
        self.module12_0 = Module12(batchnorm2d_0_num_features=64)
        self.module12_1 = Module12(batchnorm2d_0_num_features=64)
        self.conv2d_75 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module15_1 = Module15(resizebilinear_0_size=(160, 160),
                                   conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=64)
        self.module12_2 = Module12(batchnorm2d_0_num_features=64)
        self.conv2d_85 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_4 = nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_89 = nn.ReLU()
        self.module9_10 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_11 = Module9(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_12 = Module9(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_13 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module15_2 = Module15(resizebilinear_0_size=(160, 160),
                                   conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=64)
        self.module17_1 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=64)
        self.conv2d_50 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module15_3 = Module15(resizebilinear_0_size=(80, 80),
                                   conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=64)
        self.module17_2 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=64)
        self.conv2d_116 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module12_3 = Module12(batchnorm2d_0_num_features=64)
        self.module12_4 = Module12(batchnorm2d_0_num_features=64)
        self.module12_5 = Module12(batchnorm2d_0_num_features=64)
        self.module17_3 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=64)
        self.conv2d_142 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_4 = Module15(resizebilinear_0_size=(80, 80),
                                   conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=64)
        self.module12_6 = Module12(batchnorm2d_0_num_features=64)
        self.conv2d_155 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_40 = nn.Conv2d(in_channels=256,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_159 = nn.ReLU()
        self.module9_13 = Module9(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_14 = Module9(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_15 = Module9(conv2d_0_in_channels=1024,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_52 = nn.Conv2d(in_channels=256,
                                   out_channels=256,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module15_5 = Module15(resizebilinear_0_size=(80, 80),
                                   conv2d_1_in_channels=256,
                                   conv2d_1_out_channels=256)
        self.module17_4 = Module17(conv2d_1_in_channels=256, conv2d_1_out_channels=256)
        self.conv2d_118 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_6 = Module15(resizebilinear_0_size=(40, 40),
                                   conv2d_1_in_channels=256,
                                   conv2d_1_out_channels=256)
        self.module17_5 = Module17(conv2d_1_in_channels=256, conv2d_1_out_channels=256)
        self.conv2d_199 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module12_7 = Module12(batchnorm2d_0_num_features=256)
        self.module12_8 = Module12(batchnorm2d_0_num_features=256)
        self.module12_9 = Module12(batchnorm2d_0_num_features=256)
        self.module17_6 = Module17(conv2d_1_in_channels=256, conv2d_1_out_channels=256)
        self.conv2d_225 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_7 = Module15(resizebilinear_0_size=(40, 40),
                                   conv2d_1_in_channels=256,
                                   conv2d_1_out_channels=256)
        self.module12_10 = Module12(batchnorm2d_0_num_features=256)
        self.conv2d_238 = nn.Conv2d(in_channels=256,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_108 = nn.Conv2d(in_channels=512,
                                    out_channels=64,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_242 = nn.ReLU()
        self.module9_16 = Module9(conv2d_0_in_channels=512,
                                  conv2d_0_out_channels=512,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_17 = Module9(conv2d_0_in_channels=1024,
                                  conv2d_0_out_channels=512,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_18 = Module9(conv2d_0_in_channels=2048,
                                  conv2d_0_out_channels=512,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_120 = nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_8 = Module15(resizebilinear_0_size=(40, 40),
                                   conv2d_1_in_channels=512,
                                   conv2d_1_out_channels=512)
        self.module17_7 = Module17(conv2d_1_in_channels=512, conv2d_1_out_channels=512)
        self.conv2d_201 = nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_9 = Module15(resizebilinear_0_size=(20, 20),
                                   conv2d_1_in_channels=512,
                                   conv2d_1_out_channels=512)
        self.module17_8 = Module17(conv2d_1_in_channels=512, conv2d_1_out_channels=512)
        self.conv2d_255 = nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module12_11 = Module12(batchnorm2d_0_num_features=512)
        self.module12_12 = Module12(batchnorm2d_0_num_features=512)
        self.module12_13 = Module12(batchnorm2d_0_num_features=512)
        self.module17_9 = Module17(conv2d_1_in_channels=512, conv2d_1_out_channels=512)
        self.conv2d_274 = nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_10 = Module15(resizebilinear_0_size=(20, 20),
                                    conv2d_1_in_channels=512,
                                    conv2d_1_out_channels=512)
        self.module12_14 = Module12(batchnorm2d_0_num_features=512)
        self.conv2d_285 = nn.Conv2d(in_channels=512,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_191 = nn.Conv2d(in_channels=1024,
                                    out_channels=64,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_289 = nn.ReLU()
        self.module9_19 = Module9(conv2d_0_in_channels=1024,
                                  conv2d_0_out_channels=1024,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module9_20 = Module9(conv2d_0_in_channels=2048,
                                  conv2d_0_out_channels=1024,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_203 = nn.Conv2d(in_channels=1024,
                                    out_channels=1024,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_10 = Module17(conv2d_1_in_channels=1024, conv2d_1_out_channels=1024)
        self.conv2d_256 = nn.Conv2d(in_channels=1024,
                                    out_channels=1024,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_11 = Module15(resizebilinear_0_size=(20, 20),
                                    conv2d_1_in_channels=1024,
                                    conv2d_1_out_channels=1024)
        self.module12_15 = Module12(batchnorm2d_0_num_features=1024)
        self.module12_16 = Module12(batchnorm2d_0_num_features=1024)
        self.module17_11 = Module17(conv2d_1_in_channels=1024, conv2d_1_out_channels=1024)
        self.conv2d_272 = nn.Conv2d(in_channels=1024,
                                    out_channels=1024,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module12_17 = Module12(batchnorm2d_0_num_features=1024)
        self.conv2d_286 = nn.Conv2d(in_channels=1024,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_251 = nn.Conv2d(in_channels=2048,
                                    out_channels=64,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_290 = nn.ReLU()
        self.module9_21 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.pad_avgpool2d_292 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_292 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module9_22 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_295 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_12 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=32)
        self.conv2d_299 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_12 = Module15(resizebilinear_0_size=(10, 10), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_18 = Module12(batchnorm2d_0_num_features=64)
        self.module12_19 = Module12(batchnorm2d_0_num_features=32)
        self.conv2d_310 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_13 = Module15(resizebilinear_0_size=(10, 10), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_20 = Module12(batchnorm2d_0_num_features=64)
        self.module9_23 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_317 = P.ResizeBilinear(size=(20, 20), align_corners=False)
        self.module9_24 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.pad_avgpool2d_320 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_320 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module9_25 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_323 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_13 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=32)
        self.conv2d_327 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_14 = Module15(resizebilinear_0_size=(20, 20), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_21 = Module12(batchnorm2d_0_num_features=64)
        self.module12_22 = Module12(batchnorm2d_0_num_features=32)
        self.conv2d_338 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_15 = Module15(resizebilinear_0_size=(20, 20), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_23 = Module12(batchnorm2d_0_num_features=64)
        self.module9_26 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_345 = P.ResizeBilinear(size=(40, 40), align_corners=False)
        self.module9_27 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.pad_avgpool2d_348 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_348 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module9_28 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_351 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_14 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=32)
        self.conv2d_355 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_16 = Module15(resizebilinear_0_size=(40, 40), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_24 = Module12(batchnorm2d_0_num_features=64)
        self.module12_25 = Module12(batchnorm2d_0_num_features=32)
        self.conv2d_366 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_17 = Module15(resizebilinear_0_size=(40, 40), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_26 = Module12(batchnorm2d_0_num_features=64)
        self.module9_29 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_373 = P.ResizeBilinear(size=(80, 80), align_corners=False)
        self.module9_30 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.pad_avgpool2d_376 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_376 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module9_31 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_379 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_15 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=32)
        self.conv2d_383 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_18 = Module15(resizebilinear_0_size=(80, 80), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_27 = Module12(batchnorm2d_0_num_features=64)
        self.module12_28 = Module12(batchnorm2d_0_num_features=32)
        self.conv2d_394 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_19 = Module15(resizebilinear_0_size=(80, 80), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_29 = Module12(batchnorm2d_0_num_features=64)
        self.module9_32 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_401 = P.ResizeBilinear(size=(160, 160), align_corners=False)
        self.module9_33 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=64,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.pad_avgpool2d_404 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_404 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module9_34 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_407 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module17_16 = Module17(conv2d_1_in_channels=64, conv2d_1_out_channels=32)
        self.conv2d_411 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_20 = Module15(resizebilinear_0_size=(160, 160), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_30 = Module12(batchnorm2d_0_num_features=64)
        self.module12_31 = Module12(batchnorm2d_0_num_features=32)
        self.conv2d_422 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module15_21 = Module15(resizebilinear_0_size=(160, 160), conv2d_1_in_channels=32, conv2d_1_out_channels=64)
        self.module12_32 = Module12(batchnorm2d_0_num_features=64)
        self.module9_35 = Module9(conv2d_0_in_channels=64,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_429 = P.ResizeBilinear(size=(320, 320), align_corners=False)
        self.module9_36 = Module9(conv2d_0_in_channels=32,
                                  conv2d_0_out_channels=32,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_432 = nn.Conv2d(in_channels=32,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module9_0_opt = self.module9_0(opt_maxpool2d_2)
        module9_1_opt = self.module9_1(module9_0_opt)
        opt_conv2d_19 = self.conv2d_19(module9_1_opt)
        opt_conv2d_7 = self.conv2d_7(opt_maxpool2d_2)
        opt_add_20 = P.Add()(opt_conv2d_19, opt_conv2d_7)
        opt_relu_21 = self.relu_21(opt_add_20)
        module4_0_opt = self.module4_0(opt_relu_21)
        module4_1_opt = self.module4_1(module4_0_opt)
        module9_2_opt = self.module9_2(module4_1_opt)
        module9_3_opt = self.module9_3(module9_2_opt)
        opt_conv2d_61 = self.conv2d_61(module9_3_opt)
        opt_conv2d_37 = self.conv2d_37(module4_1_opt)
        opt_add_65 = P.Add()(opt_conv2d_61, opt_conv2d_37)
        opt_relu_69 = self.relu_69(opt_add_65)
        module4_2_opt = self.module4_2(opt_relu_69)
        module4_3_opt = self.module4_3(module4_2_opt)
        module4_4_opt = self.module4_4(module4_3_opt)
        module9_4_opt = self.module9_4(module4_4_opt)
        module9_5_opt = self.module9_5(module9_4_opt)
        opt_conv2d_129 = self.conv2d_129(module9_5_opt)
        opt_conv2d_105 = self.conv2d_105(module4_4_opt)
        opt_add_133 = P.Add()(opt_conv2d_129, opt_conv2d_105)
        opt_relu_137 = self.relu_137(opt_add_133)
        module4_5_opt = self.module4_5(opt_relu_137)
        module4_6_opt = self.module4_6(module4_5_opt)
        module4_7_opt = self.module4_7(module4_6_opt)
        module4_8_opt = self.module4_8(module4_7_opt)
        module4_9_opt = self.module4_9(module4_8_opt)
        module9_6_opt = self.module9_6(module4_9_opt)
        module9_7_opt = self.module9_7(module9_6_opt)
        opt_conv2d_212 = self.conv2d_212(module9_7_opt)
        opt_conv2d_188 = self.conv2d_188(module4_9_opt)
        opt_add_216 = P.Add()(opt_conv2d_212, opt_conv2d_188)
        opt_relu_220 = self.relu_220(opt_add_216)
        module4_10_opt = self.module4_10(opt_relu_220)
        module4_11_opt = self.module4_11(module4_10_opt)
        module9_8_opt = self.module9_8(opt_relu_1)
        module9_9_opt = self.module9_9(module4_1_opt)
        opt_conv2d_11 = self.conv2d_11(module9_8_opt)
        module17_0_opt = self.module17_0(module9_8_opt)
        opt_conv2d_47 = self.conv2d_47(module9_9_opt)
        module15_0_opt = self.module15_0(module9_9_opt)
        opt_add_63 = P.Add()(opt_conv2d_11, module15_0_opt)
        module12_0_opt = self.module12_0(opt_add_63)
        opt_add_55 = P.Add()(opt_conv2d_47, module17_0_opt)
        module12_1_opt = self.module12_1(opt_add_55)
        opt_conv2d_75 = self.conv2d_75(module12_0_opt)
        module15_1_opt = self.module15_1(module12_1_opt)
        opt_add_78 = P.Add()(opt_conv2d_75, module15_1_opt)
        module12_2_opt = self.module12_2(opt_add_78)
        opt_conv2d_85 = self.conv2d_85(module12_2_opt)
        opt_conv2d_4 = self.conv2d_4(opt_relu_1)
        opt_add_87 = P.Add()(opt_conv2d_85, opt_conv2d_4)
        opt_relu_89 = self.relu_89(opt_add_87)
        module9_10_opt = self.module9_10(opt_relu_1)
        module9_11_opt = self.module9_11(module4_1_opt)
        module9_12_opt = self.module9_12(module4_4_opt)
        opt_conv2d_13 = self.conv2d_13(module9_10_opt)
        module15_2_opt = self.module15_2(module9_11_opt)
        module17_1_opt = self.module17_1(module9_10_opt)
        opt_conv2d_50 = self.conv2d_50(module9_11_opt)
        module15_3_opt = self.module15_3(module9_12_opt)
        module17_2_opt = self.module17_2(module9_11_opt)
        opt_conv2d_116 = self.conv2d_116(module9_12_opt)
        opt_add_64 = P.Add()(opt_conv2d_13, module15_2_opt)
        module12_3_opt = self.module12_3(opt_add_64)
        opt_add_58 = P.Add()(module17_1_opt, opt_conv2d_50)
        opt_add_130 = P.Add()(opt_add_58, module15_3_opt)
        module12_4_opt = self.module12_4(opt_add_130)
        opt_add_124 = P.Add()(module17_2_opt, opt_conv2d_116)
        module12_5_opt = self.module12_5(opt_add_124)
        module17_3_opt = self.module17_3(module12_3_opt)
        opt_conv2d_142 = self.conv2d_142(module12_4_opt)
        module15_4_opt = self.module15_4(module12_5_opt)
        opt_add_146 = P.Add()(module17_3_opt, opt_conv2d_142)
        opt_add_149 = P.Add()(opt_add_146, module15_4_opt)
        module12_6_opt = self.module12_6(opt_add_149)
        opt_conv2d_155 = self.conv2d_155(module12_6_opt)
        opt_conv2d_40 = self.conv2d_40(module4_1_opt)
        opt_add_157 = P.Add()(opt_conv2d_155, opt_conv2d_40)
        opt_relu_159 = self.relu_159(opt_add_157)
        module9_13_opt = self.module9_13(module4_1_opt)
        module9_14_opt = self.module9_14(module4_4_opt)
        module9_15_opt = self.module9_15(module4_9_opt)
        opt_conv2d_52 = self.conv2d_52(module9_13_opt)
        module15_5_opt = self.module15_5(module9_14_opt)
        module17_4_opt = self.module17_4(module9_13_opt)
        opt_conv2d_118 = self.conv2d_118(module9_14_opt)
        module15_6_opt = self.module15_6(module9_15_opt)
        module17_5_opt = self.module17_5(module9_14_opt)
        opt_conv2d_199 = self.conv2d_199(module9_15_opt)
        opt_add_132 = P.Add()(opt_conv2d_52, module15_5_opt)
        module12_7_opt = self.module12_7(opt_add_132)
        opt_add_126 = P.Add()(module17_4_opt, opt_conv2d_118)
        opt_add_213 = P.Add()(opt_add_126, module15_6_opt)
        module12_8_opt = self.module12_8(opt_add_213)
        opt_add_207 = P.Add()(module17_5_opt, opt_conv2d_199)
        module12_9_opt = self.module12_9(opt_add_207)
        module17_6_opt = self.module17_6(module12_7_opt)
        opt_conv2d_225 = self.conv2d_225(module12_8_opt)
        module15_7_opt = self.module15_7(module12_9_opt)
        opt_add_229 = P.Add()(module17_6_opt, opt_conv2d_225)
        opt_add_232 = P.Add()(opt_add_229, module15_7_opt)
        module12_10_opt = self.module12_10(opt_add_232)
        opt_conv2d_238 = self.conv2d_238(module12_10_opt)
        opt_conv2d_108 = self.conv2d_108(module4_4_opt)
        opt_add_240 = P.Add()(opt_conv2d_238, opt_conv2d_108)
        opt_relu_242 = self.relu_242(opt_add_240)
        module9_16_opt = self.module9_16(module4_4_opt)
        module9_17_opt = self.module9_17(module4_9_opt)
        module9_18_opt = self.module9_18(module4_11_opt)
        opt_conv2d_120 = self.conv2d_120(module9_16_opt)
        module15_8_opt = self.module15_8(module9_17_opt)
        module17_7_opt = self.module17_7(module9_16_opt)
        opt_conv2d_201 = self.conv2d_201(module9_17_opt)
        module15_9_opt = self.module15_9(module9_18_opt)
        module17_8_opt = self.module17_8(module9_17_opt)
        opt_conv2d_255 = self.conv2d_255(module9_18_opt)
        opt_add_215 = P.Add()(opt_conv2d_120, module15_8_opt)
        module12_11_opt = self.module12_11(opt_add_215)
        opt_add_209 = P.Add()(module17_7_opt, opt_conv2d_201)
        opt_add_262 = P.Add()(opt_add_209, module15_9_opt)
        module12_12_opt = self.module12_12(opt_add_262)
        opt_add_259 = P.Add()(module17_8_opt, opt_conv2d_255)
        module12_13_opt = self.module12_13(opt_add_259)
        module17_9_opt = self.module17_9(module12_11_opt)
        opt_conv2d_274 = self.conv2d_274(module12_12_opt)
        module15_10_opt = self.module15_10(module12_13_opt)
        opt_add_277 = P.Add()(module17_9_opt, opt_conv2d_274)
        opt_add_279 = P.Add()(opt_add_277, module15_10_opt)
        module12_14_opt = self.module12_14(opt_add_279)
        opt_conv2d_285 = self.conv2d_285(module12_14_opt)
        opt_conv2d_191 = self.conv2d_191(module4_9_opt)
        opt_add_287 = P.Add()(opt_conv2d_285, opt_conv2d_191)
        opt_relu_289 = self.relu_289(opt_add_287)
        module9_19_opt = self.module9_19(module4_9_opt)
        module9_20_opt = self.module9_20(module4_11_opt)
        opt_conv2d_203 = self.conv2d_203(module9_19_opt)
        module17_10_opt = self.module17_10(module9_19_opt)
        opt_conv2d_256 = self.conv2d_256(module9_20_opt)
        module15_11_opt = self.module15_11(module9_20_opt)
        opt_add_265 = P.Add()(opt_conv2d_203, module15_11_opt)
        module12_15_opt = self.module12_15(opt_add_265)
        opt_add_260 = P.Add()(opt_conv2d_256, module17_10_opt)
        module12_16_opt = self.module12_16(opt_add_260)
        module17_11_opt = self.module17_11(module12_15_opt)
        opt_conv2d_272 = self.conv2d_272(module12_16_opt)
        opt_add_280 = P.Add()(module17_11_opt, opt_conv2d_272)
        module12_17_opt = self.module12_17(opt_add_280)
        opt_conv2d_286 = self.conv2d_286(module12_17_opt)
        opt_conv2d_251 = self.conv2d_251(module4_11_opt)
        opt_add_288 = P.Add()(opt_conv2d_286, opt_conv2d_251)
        opt_relu_290 = self.relu_290(opt_add_288)
        module9_21_opt = self.module9_21(opt_relu_290)
        opt_avgpool2d_292 = self.pad_avgpool2d_292(opt_relu_290)
        opt_avgpool2d_292 = self.avgpool2d_292(opt_avgpool2d_292)
        module9_22_opt = self.module9_22(opt_avgpool2d_292)
        opt_conv2d_295 = self.conv2d_295(module9_21_opt)
        module17_12_opt = self.module17_12(module9_21_opt)
        opt_conv2d_299 = self.conv2d_299(module9_22_opt)
        module15_12_opt = self.module15_12(module9_22_opt)
        opt_add_304 = P.Add()(opt_conv2d_295, module15_12_opt)
        module12_18_opt = self.module12_18(opt_add_304)
        opt_add_301 = P.Add()(opt_conv2d_299, module17_12_opt)
        module12_19_opt = self.module12_19(opt_add_301)
        opt_conv2d_310 = self.conv2d_310(module12_18_opt)
        module15_13_opt = self.module15_13(module12_19_opt)
        opt_add_311 = P.Add()(opt_conv2d_310, module15_13_opt)
        module12_20_opt = self.module12_20(opt_add_311)
        opt_add_314 = P.Add()(module12_20_opt, opt_relu_290)
        module9_23_opt = self.module9_23(opt_add_314)
        opt_resizebilinear_317 = self.resizebilinear_317(module9_23_opt)
        opt_add_318 = P.Add()(opt_relu_289, opt_resizebilinear_317)
        module9_24_opt = self.module9_24(opt_add_318)
        opt_avgpool2d_320 = self.pad_avgpool2d_320(opt_add_318)
        opt_avgpool2d_320 = self.avgpool2d_320(opt_avgpool2d_320)
        module9_25_opt = self.module9_25(opt_avgpool2d_320)
        opt_conv2d_323 = self.conv2d_323(module9_24_opt)
        module17_13_opt = self.module17_13(module9_24_opt)
        opt_conv2d_327 = self.conv2d_327(module9_25_opt)
        module15_14_opt = self.module15_14(module9_25_opt)
        opt_add_332 = P.Add()(opt_conv2d_323, module15_14_opt)
        module12_21_opt = self.module12_21(opt_add_332)
        opt_add_329 = P.Add()(opt_conv2d_327, module17_13_opt)
        module12_22_opt = self.module12_22(opt_add_329)
        opt_conv2d_338 = self.conv2d_338(module12_21_opt)
        module15_15_opt = self.module15_15(module12_22_opt)
        opt_add_339 = P.Add()(opt_conv2d_338, module15_15_opt)
        module12_23_opt = self.module12_23(opt_add_339)
        opt_add_342 = P.Add()(module12_23_opt, opt_add_318)
        module9_26_opt = self.module9_26(opt_add_342)
        opt_resizebilinear_345 = self.resizebilinear_345(module9_26_opt)
        opt_add_346 = P.Add()(opt_relu_242, opt_resizebilinear_345)
        module9_27_opt = self.module9_27(opt_add_346)
        opt_avgpool2d_348 = self.pad_avgpool2d_348(opt_add_346)
        opt_avgpool2d_348 = self.avgpool2d_348(opt_avgpool2d_348)
        module9_28_opt = self.module9_28(opt_avgpool2d_348)
        opt_conv2d_351 = self.conv2d_351(module9_27_opt)
        module17_14_opt = self.module17_14(module9_27_opt)
        opt_conv2d_355 = self.conv2d_355(module9_28_opt)
        module15_16_opt = self.module15_16(module9_28_opt)
        opt_add_360 = P.Add()(opt_conv2d_351, module15_16_opt)
        module12_24_opt = self.module12_24(opt_add_360)
        opt_add_357 = P.Add()(opt_conv2d_355, module17_14_opt)
        module12_25_opt = self.module12_25(opt_add_357)
        opt_conv2d_366 = self.conv2d_366(module12_24_opt)
        module15_17_opt = self.module15_17(module12_25_opt)
        opt_add_367 = P.Add()(opt_conv2d_366, module15_17_opt)
        module12_26_opt = self.module12_26(opt_add_367)
        opt_add_370 = P.Add()(module12_26_opt, opt_add_346)
        module9_29_opt = self.module9_29(opt_add_370)
        opt_resizebilinear_373 = self.resizebilinear_373(module9_29_opt)
        opt_add_374 = P.Add()(opt_relu_159, opt_resizebilinear_373)
        module9_30_opt = self.module9_30(opt_add_374)
        opt_avgpool2d_376 = self.pad_avgpool2d_376(opt_add_374)
        opt_avgpool2d_376 = self.avgpool2d_376(opt_avgpool2d_376)
        module9_31_opt = self.module9_31(opt_avgpool2d_376)
        opt_conv2d_379 = self.conv2d_379(module9_30_opt)
        module17_15_opt = self.module17_15(module9_30_opt)
        opt_conv2d_383 = self.conv2d_383(module9_31_opt)
        module15_18_opt = self.module15_18(module9_31_opt)
        opt_add_388 = P.Add()(opt_conv2d_379, module15_18_opt)
        module12_27_opt = self.module12_27(opt_add_388)
        opt_add_385 = P.Add()(opt_conv2d_383, module17_15_opt)
        module12_28_opt = self.module12_28(opt_add_385)
        opt_conv2d_394 = self.conv2d_394(module12_27_opt)
        module15_19_opt = self.module15_19(module12_28_opt)
        opt_add_395 = P.Add()(opt_conv2d_394, module15_19_opt)
        module12_29_opt = self.module12_29(opt_add_395)
        opt_add_398 = P.Add()(module12_29_opt, opt_add_374)
        module9_32_opt = self.module9_32(opt_add_398)
        opt_resizebilinear_401 = self.resizebilinear_401(module9_32_opt)
        opt_add_402 = P.Add()(opt_relu_89, opt_resizebilinear_401)
        module9_33_opt = self.module9_33(opt_add_402)
        opt_avgpool2d_404 = self.pad_avgpool2d_404(opt_add_402)
        opt_avgpool2d_404 = self.avgpool2d_404(opt_avgpool2d_404)
        module9_34_opt = self.module9_34(opt_avgpool2d_404)
        opt_conv2d_407 = self.conv2d_407(module9_33_opt)
        module17_16_opt = self.module17_16(module9_33_opt)
        opt_conv2d_411 = self.conv2d_411(module9_34_opt)
        module15_20_opt = self.module15_20(module9_34_opt)
        opt_add_416 = P.Add()(opt_conv2d_407, module15_20_opt)
        module12_30_opt = self.module12_30(opt_add_416)
        opt_add_413 = P.Add()(opt_conv2d_411, module17_16_opt)
        module12_31_opt = self.module12_31(opt_add_413)
        opt_conv2d_422 = self.conv2d_422(module12_30_opt)
        module15_21_opt = self.module15_21(module12_31_opt)
        opt_add_423 = P.Add()(opt_conv2d_422, module15_21_opt)
        module12_32_opt = self.module12_32(opt_add_423)
        opt_add_426 = P.Add()(module12_32_opt, opt_add_402)
        module9_35_opt = self.module9_35(opt_add_426)
        opt_resizebilinear_429 = self.resizebilinear_429(module9_35_opt)
        module9_36_opt = self.module9_36(opt_resizebilinear_429)
        opt_conv2d_432 = self.conv2d_432(module9_36_opt)
        return opt_conv2d_432
