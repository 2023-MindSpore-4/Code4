import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module0, self).__init__()
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


class Module18(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size, module0_0_conv2d_0_stride,
                 module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_0_kernel_size, module0_1_conv2d_0_stride,
                 module0_1_conv2d_0_padding, module0_1_conv2d_0_pad_mode):
        super(Module18, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_1_conv2d_0_stride,
                                 conv2d_0_padding=module0_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_1_opt)
        return opt_conv2d_0


class Module20(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_3_in_channels, conv2d_3_out_channels,
                 module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size,
                 module0_0_conv2d_0_stride, module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_0_kernel_size,
                 module0_1_conv2d_0_stride, module0_1_conv2d_0_padding, module0_1_conv2d_0_pad_mode,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_0_kernel_size,
                 module0_2_conv2d_0_stride, module0_2_conv2d_0_padding, module0_2_conv2d_0_pad_mode,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_0_kernel_size,
                 module0_3_conv2d_0_stride, module0_3_conv2d_0_padding, module0_3_conv2d_0_pad_mode):
        super(Module20, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_1_conv2d_0_stride,
                                 conv2d_0_padding=module0_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_2_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_2_conv2d_0_stride,
                                 conv2d_0_padding=module0_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_2_conv2d_0_pad_mode)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_3_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_3_conv2d_0_stride,
                                 conv2d_0_padding=module0_3_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_3_conv2d_0_pad_mode)
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_1_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, x)
        opt_relu_2 = self.relu_2(opt_add_1)
        module0_2_opt = self.module0_2(opt_relu_2)
        module0_3_opt = self.module0_3(module0_2_opt)
        opt_conv2d_3 = self.conv2d_3(module0_3_opt)
        opt_add_4 = P.Add()(opt_conv2d_3, opt_relu_2)
        opt_relu_5 = self.relu_5(opt_add_4)
        return opt_relu_5


class Module19(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size, module0_0_conv2d_0_stride,
                 module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_0_kernel_size, module0_1_conv2d_0_stride,
                 module0_1_conv2d_0_padding, module0_1_conv2d_0_pad_mode):
        super(Module19, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_1_conv2d_0_stride,
                                 conv2d_0_padding=module0_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_1_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, x)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class Module16(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_kernel_size, conv2d_0_padding, conv2d_0_pad_mode,
                 module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size,
                 module0_0_conv2d_0_stride, module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode):
        super(Module16, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=256,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=(1, 1),
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_1 = nn.Sigmoid()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_sigmoid_1 = self.sigmoid_1(opt_conv2d_0)
        return opt_sigmoid_1


class Module8(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size,
                 module0_0_conv2d_0_stride, module0_0_conv2d_0_padding, module0_0_conv2d_0_pad_mode):
        super(Module8, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module0_0_conv2d_0_stride,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode)
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=(9, 9))
        self.squeeze_2 = P.Squeeze(axis=(2, 3))
        self.pad_maxpool2d_1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(9, 9), stride=(9, 9))
        self.squeeze_3 = P.Squeeze(axis=(2, 3))
        self.matmul_4_w = Parameter(Tensor(np.random.uniform(0, 1, (256, 64)).astype(np.float32)), name=None)
        self.batchnorm1d_6 = nn.BatchNorm1d(num_features=64, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_8 = nn.ReLU()
        self.matmul_10_w = Parameter(Tensor(np.random.uniform(0, 1, (64, 256)).astype(np.float32)), name=None)
        self.batchnorm1d_12 = nn.BatchNorm1d(num_features=256, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.matmul_5_w = Parameter(Tensor(np.random.uniform(0, 1, (256, 64)).astype(np.float32)), name=None)
        self.batchnorm1d_7 = nn.BatchNorm1d(num_features=64, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_9 = nn.ReLU()
        self.matmul_11_w = Parameter(Tensor(np.random.uniform(0, 1, (64, 256)).astype(np.float32)), name=None)
        self.batchnorm1d_13 = nn.BatchNorm1d(num_features=256, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.sigmoid_15 = nn.Sigmoid()
        self.expanddims_16 = P.ExpandDims()
        self.expanddims_16_axis = -1
        self.expanddims_17 = P.ExpandDims()
        self.expanddims_17_axis = -1

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_avgpool2d_0 = self.avgpool2d_0(module0_0_opt)
        opt_squeeze_2 = self.squeeze_2(opt_avgpool2d_0)
        opt_maxpool2d_1 = self.pad_maxpool2d_1(module0_0_opt)
        opt_maxpool2d_1 = self.maxpool2d_1(opt_maxpool2d_1)
        opt_squeeze_3 = self.squeeze_3(opt_maxpool2d_1)
        opt_matmul_4 = P.matmul(opt_squeeze_2, self.matmul_4_w)
        opt_batchnorm1d_6 = self.batchnorm1d_6(opt_matmul_4)
        opt_relu_8 = self.relu_8(opt_batchnorm1d_6)
        opt_matmul_10 = P.matmul(opt_relu_8, self.matmul_10_w)
        opt_batchnorm1d_12 = self.batchnorm1d_12(opt_matmul_10)
        opt_matmul_5 = P.matmul(opt_squeeze_3, self.matmul_5_w)
        opt_batchnorm1d_7 = self.batchnorm1d_7(opt_matmul_5)
        opt_relu_9 = self.relu_9(opt_batchnorm1d_7)
        opt_matmul_11 = P.matmul(opt_relu_9, self.matmul_11_w)
        opt_batchnorm1d_13 = self.batchnorm1d_13(opt_matmul_11)
        opt_add_14 = P.Add()(opt_batchnorm1d_12, opt_batchnorm1d_13)
        opt_sigmoid_15 = self.sigmoid_15(opt_add_14)
        opt_expanddims_16 = self.expanddims_16(opt_sigmoid_15, self.expanddims_16_axis)
        opt_expanddims_17 = self.expanddims_17(opt_expanddims_16, self.expanddims_17_axis)
        opt_mul_18 = P.Mul()(module0_0_opt, opt_expanddims_17)
        opt_add_19 = P.Add()(opt_mul_18, module0_0_opt)
        return opt_add_19


class Module14(nn.Cell):
    def __init__(self):
        super(Module14, self).__init__()
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


class WSLNet_down(nn.Cell):
    def __init__(self):
        super(WSLNet_down, self).__init__()
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
        self.module18_0 = Module18(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_in_channels=64,
                                   module0_0_conv2d_0_out_channels=64,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=64,
                                   module0_1_conv2d_0_out_channels=64,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.conv2d_4 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_10 = nn.ReLU()
        self.module20_0 = Module20(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   conv2d_3_in_channels=64,
                                   conv2d_3_out_channels=256,
                                   module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=64,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=64,
                                   module0_1_conv2d_0_out_channels=64,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad",
                                   module0_2_conv2d_0_in_channels=256,
                                   module0_2_conv2d_0_out_channels=64,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_stride=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_3_conv2d_0_in_channels=64,
                                   module0_3_conv2d_0_out_channels=64,
                                   module0_3_conv2d_0_kernel_size=(3, 3),
                                   module0_3_conv2d_0_stride=(1, 1),
                                   module0_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_3_conv2d_0_pad_mode="pad")
        self.module18_1 = Module18(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=128,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(2, 2),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.conv2d_26 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_32 = nn.ReLU()
        self.module20_1 = Module20(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   conv2d_3_in_channels=128,
                                   conv2d_3_out_channels=512,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=128,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad",
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=128,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_stride=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_3_conv2d_0_in_channels=128,
                                   module0_3_conv2d_0_out_channels=128,
                                   module0_3_conv2d_0_kernel_size=(3, 3),
                                   module0_3_conv2d_0_stride=(1, 1),
                                   module0_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_3_conv2d_0_pad_mode="pad")
        self.module19_0 = Module19(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=128,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.module18_2 = Module18(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(2, 2),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.conv2d_55 = nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_61 = nn.ReLU()
        self.module20_2 = Module20(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   conv2d_3_in_channels=256,
                                   conv2d_3_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad",
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_stride=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_3_conv2d_0_in_channels=256,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_0_kernel_size=(3, 3),
                                   module0_3_conv2d_0_stride=(1, 1),
                                   module0_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_3_conv2d_0_pad_mode="pad")
        self.module20_3 = Module20(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   conv2d_3_in_channels=256,
                                   conv2d_3_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad",
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_stride=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_3_conv2d_0_in_channels=256,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_0_kernel_size=(3, 3),
                                   module0_3_conv2d_0_stride=(1, 1),
                                   module0_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_3_conv2d_0_pad_mode="pad")
        self.module19_1 = Module19(conv2d_0_in_channels=256,
                                   conv2d_0_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.module18_3 = Module18(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(2, 2),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad")
        self.conv2d_98 = nn.Conv2d(in_channels=1024,
                                   out_channels=2048,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_104 = nn.ReLU()
        self.module20_4 = Module20(conv2d_0_in_channels=512,
                                   conv2d_0_out_channels=2048,
                                   conv2d_3_in_channels=512,
                                   conv2d_3_out_channels=2048,
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_0_kernel_size=(3, 3),
                                   module0_1_conv2d_0_stride=(1, 1),
                                   module0_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_1_conv2d_0_pad_mode="pad",
                                   module0_2_conv2d_0_in_channels=2048,
                                   module0_2_conv2d_0_out_channels=512,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_stride=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_3_conv2d_0_in_channels=512,
                                   module0_3_conv2d_0_out_channels=512,
                                   module0_3_conv2d_0_kernel_size=(3, 3),
                                   module0_3_conv2d_0_stride=(1, 1),
                                   module0_3_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_3_conv2d_0_pad_mode="pad")
        self.module0_0 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.reducemean_120 = P.ReduceMean(keep_dims=True)
        self.reducemean_120_axis = (2, 3)
        self.module16_0 = Module16(conv2d_0_in_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid")
        self.module0_1 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.reducemean_122 = P.ReduceMean(keep_dims=True)
        self.reducemean_122_axis = (2, 3)
        self.module16_1 = Module16(conv2d_0_in_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid")
        self.module0_2 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.reducemean_124 = P.ReduceMean(keep_dims=True)
        self.reducemean_124_axis = (2, 3)
        self.module16_2 = Module16(conv2d_0_in_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid")
        self.module0_3 = Module0(conv2d_0_in_channels=2048,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_126 = nn.Conv2d(in_channels=2048,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_135 = P.StridedSlice()
        self.stridedslice_135_begin = (0, 0, 0, 0)
        self.stridedslice_135_end = (2, 256, 9, 9)
        self.stridedslice_135_strides = (1, 1, 1, 1)
        self.stridedslice_136 = P.StridedSlice()
        self.stridedslice_136_begin = (0, 256, 0, 0)
        self.stridedslice_136_end = (2, 512, 9, 9)
        self.stridedslice_136_strides = (1, 1, 1, 1)
        self.relu_151 = nn.ReLU()
        self.module0_4 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid")
        self.reducemean_127 = P.ReduceMean(keep_dims=True)
        self.reducemean_127_axis = (2, 3)
        self.module16_3 = Module16(conv2d_0_in_channels=256,
                                   conv2d_0_kernel_size=(1, 1),
                                   conv2d_0_padding=0,
                                   conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid")
        self.module0_5 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.conv2d_182 = nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_189 = P.StridedSlice()
        self.stridedslice_189_begin = (0, 0, 0, 0)
        self.stridedslice_189_end = (2, 256, 9, 9)
        self.stridedslice_189_strides = (1, 1, 1, 1)
        self.stridedslice_190 = P.StridedSlice()
        self.stridedslice_190_begin = (0, 256, 0, 0)
        self.stridedslice_190_end = (2, 512, 9, 9)
        self.stridedslice_190_strides = (1, 1, 1, 1)
        self.relu_220 = nn.ReLU()
        self.module0_6 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module0_7 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.resizebilinear_165 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.concat_169 = P.Concat(axis=1)
        self.module8_0 = Module8(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid")
        self.module0_8 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module16_4 = Module16(conv2d_0_in_channels=128,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module0_0_conv2d_0_in_channels=128,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(3, 3),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_0_conv2d_0_pad_mode="pad")
        self.sub_287_bias = 1.0
        self.conv2d_243 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_298 = nn.ReLU()
        self.module14_0 = Module14()
        self.module0_9 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=256,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad")
        self.module0_10 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_307 = nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_308 = P.StridedSlice()
        self.stridedslice_308_begin = (0, 0, 0, 0)
        self.stridedslice_308_end = (2, 256, 9, 9)
        self.stridedslice_308_strides = (1, 1, 1, 1)
        self.stridedslice_309 = P.StridedSlice()
        self.stridedslice_309_begin = (0, 256, 0, 0)
        self.stridedslice_309_end = (2, 512, 9, 9)
        self.stridedslice_309_strides = (1, 1, 1, 1)
        self.relu_312 = nn.ReLU()
        self.module0_11 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module0_12 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_166 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.concat_171 = P.Concat(axis=1)
        self.module8_1 = Module8(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid")
        self.module0_13 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module16_5 = Module16(conv2d_0_in_channels=128,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module0_0_conv2d_0_in_channels=128,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(3, 3),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_0_conv2d_0_pad_mode="pad")
        self.sub_289_bias = 1.0
        self.conv2d_317 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_299 = nn.ReLU()
        self.module14_1 = Module14()
        self.module0_14 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module0_15 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_324 = nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_325 = P.StridedSlice()
        self.stridedslice_325_begin = (0, 0, 0, 0)
        self.stridedslice_325_end = (2, 256, 9, 9)
        self.stridedslice_325_strides = (1, 1, 1, 1)
        self.stridedslice_326 = P.StridedSlice()
        self.stridedslice_326_begin = (0, 256, 0, 0)
        self.stridedslice_326_end = (2, 512, 9, 9)
        self.stridedslice_326_strides = (1, 1, 1, 1)
        self.relu_329 = nn.ReLU()
        self.module0_16 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module0_17 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.resizebilinear_167 = P.ResizeBilinear(size=(9, 9), align_corners=False)
        self.concat_173 = P.Concat(axis=1)
        self.module8_2 = Module8(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_0_kernel_size=(1, 1),
                                 module0_0_conv2d_0_stride=(1, 1),
                                 module0_0_conv2d_0_padding=0,
                                 module0_0_conv2d_0_pad_mode="valid")
        self.module0_18 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=128,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module16_6 = Module16(conv2d_0_in_channels=128,
                                   conv2d_0_kernel_size=(3, 3),
                                   conv2d_0_padding=(1, 1, 1, 1),
                                   conv2d_0_pad_mode="pad",
                                   module0_0_conv2d_0_in_channels=128,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(3, 3),
                                   module0_0_conv2d_0_stride=(1, 1),
                                   module0_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module0_0_conv2d_0_pad_mode="pad")
        self.sub_291_bias = 1.0
        self.conv2d_334 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_300 = nn.ReLU()
        self.module14_2 = Module14()
        self.module0_19 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.module0_20 = Module0(conv2d_0_in_channels=256,
                                  conv2d_0_out_channels=256,
                                  conv2d_0_kernel_size=(3, 3),
                                  conv2d_0_stride=(1, 1),
                                  conv2d_0_padding=(1, 1, 1, 1),
                                  conv2d_0_pad_mode="pad")
        self.conv2d_341 = nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.stridedslice_342 = P.StridedSlice()
        self.stridedslice_342_begin = (0, 0, 0, 0)
        self.stridedslice_342_end = (2, 256, 9, 9)
        self.stridedslice_342_strides = (1, 1, 1, 1)
        self.stridedslice_343 = P.StridedSlice()
        self.stridedslice_343_begin = (0, 256, 0, 0)
        self.stridedslice_343_end = (2, 512, 9, 9)
        self.stridedslice_343_strides = (1, 1, 1, 1)
        self.relu_346 = nn.ReLU()
        self.conv2d_228 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_236 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_314 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_316 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_331 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_333 = P.ResizeBilinear(size=(288, 288), align_corners=False)
        self.conv2d_347 = nn.Conv2d(in_channels=256,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.resizebilinear_348 = P.ResizeBilinear(size=(288, 288), align_corners=False)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module18_0_opt = self.module18_0(opt_maxpool2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_maxpool2d_2)
        opt_add_9 = P.Add()(module18_0_opt, opt_conv2d_4)
        opt_relu_10 = self.relu_10(opt_add_9)
        module20_0_opt = self.module20_0(opt_relu_10)
        module18_1_opt = self.module18_1(module20_0_opt)
        opt_conv2d_26 = self.conv2d_26(module20_0_opt)
        opt_add_31 = P.Add()(module18_1_opt, opt_conv2d_26)
        opt_relu_32 = self.relu_32(opt_add_31)
        module20_1_opt = self.module20_1(opt_relu_32)
        module19_0_opt = self.module19_0(module20_1_opt)
        module18_2_opt = self.module18_2(module19_0_opt)
        opt_conv2d_55 = self.conv2d_55(module19_0_opt)
        opt_add_60 = P.Add()(module18_2_opt, opt_conv2d_55)
        opt_relu_61 = self.relu_61(opt_add_60)
        module20_2_opt = self.module20_2(opt_relu_61)
        module20_3_opt = self.module20_3(module20_2_opt)
        module19_1_opt = self.module19_1(module20_3_opt)
        module18_3_opt = self.module18_3(module19_1_opt)
        opt_conv2d_98 = self.conv2d_98(module19_1_opt)
        opt_add_103 = P.Add()(module18_3_opt, opt_conv2d_98)
        opt_relu_104 = self.relu_104(opt_add_103)
        module20_4_opt = self.module20_4(opt_relu_104)
        module0_0_opt = self.module0_0(module20_4_opt)
        opt_reducemean_120 = self.reducemean_120(module20_4_opt, self.reducemean_120_axis)
        module16_0_opt = self.module16_0(opt_reducemean_120)
        opt_mul_153 = P.Mul()(module0_0_opt, module16_0_opt)
        module0_1_opt = self.module0_1(module20_4_opt)
        opt_reducemean_122 = self.reducemean_122(module20_4_opt, self.reducemean_122_axis)
        module16_1_opt = self.module16_1(opt_reducemean_122)
        opt_mul_154 = P.Mul()(module0_1_opt, module16_1_opt)
        module0_2_opt = self.module0_2(module20_4_opt)
        opt_reducemean_124 = self.reducemean_124(module20_4_opt, self.reducemean_124_axis)
        module16_2_opt = self.module16_2(opt_reducemean_124)
        opt_mul_155 = P.Mul()(module0_2_opt, module16_2_opt)
        module0_3_opt = self.module0_3(module20_4_opt)
        opt_conv2d_126 = self.conv2d_126(module20_4_opt)
        opt_stridedslice_135 = self.stridedslice_135(opt_conv2d_126, self.stridedslice_135_begin,
                                                     self.stridedslice_135_end, self.stridedslice_135_strides)
        opt_stridedslice_136 = self.stridedslice_136(opt_conv2d_126, self.stridedslice_136_begin,
                                                     self.stridedslice_136_end, self.stridedslice_136_strides)
        opt_mul_141 = P.Mul()(opt_stridedslice_135, module0_3_opt)
        opt_add_146 = P.Add()(opt_mul_141, opt_stridedslice_136)
        opt_relu_151 = self.relu_151(opt_add_146)
        module0_4_opt = self.module0_4(opt_relu_151)
        opt_reducemean_127 = self.reducemean_127(module20_4_opt, self.reducemean_127_axis)
        module16_3_opt = self.module16_3(opt_reducemean_127)
        opt_mul_164 = P.Mul()(module0_4_opt, module16_3_opt)
        module0_5_opt = self.module0_5(opt_mul_164)
        opt_conv2d_182 = self.conv2d_182(module0_5_opt)
        opt_stridedslice_189 = self.stridedslice_189(opt_conv2d_182, self.stridedslice_189_begin,
                                                     self.stridedslice_189_end, self.stridedslice_189_strides)
        opt_stridedslice_190 = self.stridedslice_190(opt_conv2d_182, self.stridedslice_190_begin,
                                                     self.stridedslice_190_end, self.stridedslice_190_strides)
        opt_mul_200 = P.Mul()(opt_stridedslice_189, module0_5_opt)
        opt_add_210 = P.Add()(opt_mul_200, opt_stridedslice_190)
        opt_relu_220 = self.relu_220(opt_add_210)
        module0_6_opt = self.module0_6(opt_relu_220)
        module0_7_opt = self.module0_7(opt_mul_153)
        opt_resizebilinear_165 = self.resizebilinear_165(module0_7_opt)
        opt_concat_169 = self.concat_169((module0_7_opt, opt_resizebilinear_165, ))
        module8_0_opt = self.module8_0(opt_concat_169)
        module0_8_opt = self.module0_8(module8_0_opt)
        module16_4_opt = self.module16_4(module0_8_opt)
        opt_mul_286 = P.Mul()(module0_7_opt, module16_4_opt)
        opt_sub_287 = self.sub_287_bias - module16_4_opt
        opt_mul_292 = P.Mul()(opt_resizebilinear_165, opt_sub_287)
        opt_add_295 = P.Add()(opt_mul_286, opt_mul_292)
        opt_conv2d_243 = self.conv2d_243(module0_6_opt)
        opt_relu_298 = self.relu_298(opt_add_295)
        opt_mul_301 = P.Mul()(opt_conv2d_243, opt_relu_298)
        module14_0_opt = self.module14_0(opt_resizebilinear_165)
        opt_add_302 = P.Add()(module14_0_opt, opt_mul_301)
        module0_9_opt = self.module0_9(opt_add_302)
        module0_10_opt = self.module0_10(module0_9_opt)
        opt_conv2d_307 = self.conv2d_307(module0_10_opt)
        opt_stridedslice_308 = self.stridedslice_308(opt_conv2d_307, self.stridedslice_308_begin,
                                                     self.stridedslice_308_end, self.stridedslice_308_strides)
        opt_stridedslice_309 = self.stridedslice_309(opt_conv2d_307, self.stridedslice_309_begin,
                                                     self.stridedslice_309_end, self.stridedslice_309_strides)
        opt_mul_310 = P.Mul()(opt_stridedslice_308, module0_10_opt)
        opt_add_311 = P.Add()(opt_mul_310, opt_stridedslice_309)
        opt_relu_312 = self.relu_312(opt_add_311)
        module0_11_opt = self.module0_11(opt_relu_312)
        module0_12_opt = self.module0_12(opt_mul_154)
        opt_resizebilinear_166 = self.resizebilinear_166(module0_12_opt)
        opt_concat_171 = self.concat_171((module0_12_opt, opt_resizebilinear_166, ))
        module8_1_opt = self.module8_1(opt_concat_171)
        module0_13_opt = self.module0_13(module8_1_opt)
        module16_5_opt = self.module16_5(module0_13_opt)
        opt_mul_288 = P.Mul()(module0_12_opt, module16_5_opt)
        opt_sub_289 = self.sub_289_bias - module16_5_opt
        opt_mul_293 = P.Mul()(opt_resizebilinear_166, opt_sub_289)
        opt_add_296 = P.Add()(opt_mul_288, opt_mul_293)
        opt_conv2d_317 = self.conv2d_317(module0_11_opt)
        opt_relu_299 = self.relu_299(opt_add_296)
        opt_mul_318 = P.Mul()(opt_conv2d_317, opt_relu_299)
        module14_1_opt = self.module14_1(opt_resizebilinear_166)
        opt_add_319 = P.Add()(module14_1_opt, opt_mul_318)
        module0_14_opt = self.module0_14(opt_add_319)
        module0_15_opt = self.module0_15(module0_14_opt)
        opt_conv2d_324 = self.conv2d_324(module0_15_opt)
        opt_stridedslice_325 = self.stridedslice_325(opt_conv2d_324, self.stridedslice_325_begin,
                                                     self.stridedslice_325_end, self.stridedslice_325_strides)
        opt_stridedslice_326 = self.stridedslice_326(opt_conv2d_324, self.stridedslice_326_begin,
                                                     self.stridedslice_326_end, self.stridedslice_326_strides)
        opt_mul_327 = P.Mul()(opt_stridedslice_325, module0_15_opt)
        opt_add_328 = P.Add()(opt_mul_327, opt_stridedslice_326)
        opt_relu_329 = self.relu_329(opt_add_328)
        module0_16_opt = self.module0_16(opt_relu_329)
        module0_17_opt = self.module0_17(opt_mul_155)
        opt_resizebilinear_167 = self.resizebilinear_167(module0_17_opt)
        opt_concat_173 = self.concat_173((module0_17_opt, opt_resizebilinear_167, ))
        module8_2_opt = self.module8_2(opt_concat_173)
        module0_18_opt = self.module0_18(module8_2_opt)
        module16_6_opt = self.module16_6(module0_18_opt)
        opt_mul_290 = P.Mul()(module0_17_opt, module16_6_opt)
        opt_sub_291 = self.sub_291_bias - module16_6_opt
        opt_mul_294 = P.Mul()(opt_resizebilinear_167, opt_sub_291)
        opt_add_297 = P.Add()(opt_mul_290, opt_mul_294)
        opt_conv2d_334 = self.conv2d_334(module0_16_opt)
        opt_relu_300 = self.relu_300(opt_add_297)
        opt_mul_335 = P.Mul()(opt_conv2d_334, opt_relu_300)
        module14_2_opt = self.module14_2(opt_resizebilinear_167)
        opt_add_336 = P.Add()(module14_2_opt, opt_mul_335)
        module0_19_opt = self.module0_19(opt_add_336)
        module0_20_opt = self.module0_20(module0_19_opt)
        opt_conv2d_341 = self.conv2d_341(module0_20_opt)
        opt_stridedslice_342 = self.stridedslice_342(opt_conv2d_341, self.stridedslice_342_begin,
                                                     self.stridedslice_342_end, self.stridedslice_342_strides)
        opt_stridedslice_343 = self.stridedslice_343(opt_conv2d_341, self.stridedslice_343_begin,
                                                     self.stridedslice_343_end, self.stridedslice_343_strides)
        opt_mul_344 = P.Mul()(opt_stridedslice_342, module0_20_opt)
        opt_add_345 = P.Add()(opt_mul_344, opt_stridedslice_343)
        opt_relu_346 = self.relu_346(opt_add_345)
        opt_conv2d_228 = self.conv2d_228(opt_relu_220)
        opt_resizebilinear_236 = self.resizebilinear_236(opt_conv2d_228)
        opt_conv2d_314 = self.conv2d_314(opt_relu_312)
        opt_resizebilinear_316 = self.resizebilinear_316(opt_conv2d_314)
        opt_conv2d_331 = self.conv2d_331(opt_relu_329)
        opt_resizebilinear_333 = self.resizebilinear_333(opt_conv2d_331)
        opt_conv2d_347 = self.conv2d_347(opt_relu_346)
        opt_resizebilinear_348 = self.resizebilinear_348(opt_conv2d_347)
        return opt_maxpool2d_2, module20_0_opt, module19_0_opt, module19_1_opt, module20_4_opt, opt_resizebilinear_236, opt_resizebilinear_316, opt_resizebilinear_333, opt_resizebilinear_348
