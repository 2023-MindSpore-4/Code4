import numpy as np
import torch
import mindspore as ms
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="pad", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


class BasicConv(nn.Cell):
    """
    BasicConv
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation, has_bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        """
        construct
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BackBone(nn.Cell):
    """
    Alexnet
    """
    def __init__(self):
        super(BackBone, self).__init__()
        self.C1 = nn.SequentialCell(
                    BasicConv(3, 64, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(64, 64, 3, stride=1,padding=1,relu=True, bn=True, bias=True)) 
        self.C2 = nn.SequentialCell(
                    BasicConv(64, 128, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(128, 128, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")) 

        self.C3 = nn.SequentialCell(
                    BasicConv(128, 256, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(256, 256, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(256, 256, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")) 

        self.C4 = nn.SequentialCell(
                    BasicConv(256, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(512, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(512, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")) 
        self.C5 = nn.SequentialCell(
                    BasicConv(512, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(512, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True),
                    BasicConv(512, 512, 3, stride=1,padding=1,relu=True, bn=True, bias=True))


    def construct(self, x):
        """define network"""
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

def check_res(pth_path, ckpt_path):
    inp = np.random.uniform(-1, 1, (4, 3, 224, 224)).astype(np.float32)
    # 注意做单元测试时，需要给Cell打训练或推理的标签
    ms_resnet = BackBone().set_train(False)
    # param_dict = ms.load_checkpoint(ckpt_path)
    # ms.load_param_into_net(ms_resnet, param_dict)
    # ms_resnet = ms.Model(ms_resnet)


    pt_resnet =torch.load(pth_path, map_location='cpu').eval()
    # pt_resnet.load_state_dict(torch.load(pth_path, map_location='cpu'))
    ms.load_checkpoint(ckpt_path, ms_resnet)
    print("========= pt_resnet conv1.weight ==========")
    print(pt_resnet.state_dict()['C1.0.weight'].detach().numpy().reshape((-1,))[:10])
    print("========= ms_resnet conv1.weight ==========")
    # print(ms_resnet)
    print(ms_resnet.C1[0].conv.weight.data.asnumpy().reshape((-1,))[:10])
    pt_res = pt_resnet(torch.from_numpy(inp))
    ms_res = ms_resnet(ms.Tensor(inp))
    print("========= pt_resnet res ==========")
    print(pt_res)
    print("========= ms_resnet res ==========")
    print(ms_res)
    print("diff", np.max(np.abs(pt_res.detach().numpy() - ms_res.asnumpy())))

pth_path = "/mnt/sde/yhw/mindspore/20230529/GLNet_TCYB2022-main/GLNet/Checkpoints/warehouse/backbone_v.pth"
ckpt_path = "/mnt/sde/yhw/mindspore/20230529/GLNet_TCYB2022-main/GLNet_MS/Checkpoints/warehouse/backbone.ckpt"
check_res(pth_path, ckpt_path)


# export LD_LIBRARY_PATH=/mnt/sdb/software/anaconda3/lib/