import mindspore
import mindspore.nn as nn
import math
import numpy as np
from mindspore import nn, dataset, context
from mindspore.common.initializer import initializer,HeNormal, Normal,TruncatedNormal
from typing import Type, Union, List, Optional
from mindspore import nn, ops
from mindspore.common.initializer import Normal
from mindspore import load_checkpoint, load_param_into_net

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)
weight_init2 = Normal(mean=0, sigma=1)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class ResidualBlockBase(nn.Cell):
    expansion: int = 1

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlockBase, self).__init__()
        if not norm:
            self.norm = nn.BatchNorm2d(out_channel)
        else:
            self.norm = norm

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, weight_init=weight_init)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x  # shortcuts

        out = self.conv1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, weight_init=weight_init)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, weight_init=weight_init)
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):

        identity = x  # shortscuts

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

def make_layer(last_out_channel, block: Type[Union[ResidualBlockBase, ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None  # shortcuts

    if stride != 1 or last_out_channel != channel * block.expansion:

        down_sample = nn.SequentialCell([
            nn.Conv2d(last_out_channel, channel * block.expansion,
                      kernel_size=1, stride=stride, weight_init=weight_init),
            nn.BatchNorm2d(channel * block.expansion, gamma_init=gamma_init)
        ])

    layers = []
    layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample))

    in_channel = channel * block.expansion

    for _ in range(1, block_nums):

        layers.append(block(in_channel, channel))

    return nn.SequentialCell(layers)


class ResNet(nn.Cell):
    def __init__(self, mode='rgb',) -> None:
        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
        if (mode == 'rgb'):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
        elif (mode == 'rgbt'):
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, weight_init=weight_init2)
        elif (mode == "share"):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
            self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, weight_init=weight_init)
        else:
            raise

        self.norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = make_layer(64, ResidualBlock, 64, 3)
        self.layer2 = make_layer(64 * ResidualBlock.expansion, ResidualBlock, 128, 4, stride=2)
        self.layer3 = make_layer(128 * ResidualBlock.expansion, ResidualBlock, 256, 6, stride=2)
        self.layer4 = make_layer(256 * ResidualBlock.expansion, ResidualBlock, 512, 3, stride=2)

    def construct(self, x):

        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(pretrained: bool, pretrianed_ckpt: str,):
    model = ResNet()

    if pretrained:
        param_dict = load_checkpoint(pretrianed_ckpt)
        load_param_into_net(model, param_dict)

    return model


def resnet50(pretrained: bool = False):
    "ResNet50模型"
    # resnet50_url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/models/application/resnet50_224_new.ckpt"
    resnet50_ckpt = "../convert/checkpoints/resnet50_224_new.ckpt"
    return _resnet(pretrained, resnet50_ckpt)

if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.ones([2, 3, 256, 256])
    x = mindspore.Tensor(x, mindspore.float32)
    net = resnet50(pretrained=True)
    print(x)
    out = net(x)
    print(out)
    print(out.shape)
