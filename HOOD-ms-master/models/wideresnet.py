import logging
import mindspore as ms
from mindspore import nn,ops
from mindspore.common.initializer import initializer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from easydl import GradientReverseModule, aToBSheduler

logger = logging.getLogger(__name__)

def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * ops.Tanh(ops.Softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def construct(self, x):
        return super().construct(x) + self.alpha


class BasicBlock(nn.Cell):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(alpha=0.1)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, has_bias=False,pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(alpha=0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, has_bias=False,pad_mode='pad')
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0,has_bias=False,pad_mode='pad') or None
        self.activate_before_residual = activate_before_residual

    def construct(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = nn.Dropout(keep_prob=self.drop_rate)
        out = self.conv2(out)
        return ops.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Cell):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layer(x)



class WideResNet(nn.Cell):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, has_bias=False,pad_mode='pad')
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(alpha=0.1)
        # self.fc = nn.Linear(channels[3], num_classes)
        self.feature_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='leaky_relu'),
                    m.weight.shape, m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones", m.gamma.shape, m.gamma.dtype))
                m.beta.set_data(initializer("zeros", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(
                    ms.common.initializer.XavierNormal(),
                    m.weight.shape, m.weight.dtype))
                m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))


    def construct(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d(out, 1)
        out = out.view(-1, self.feature_dim)
        return out

    def update_batch_stats(self, flag):
        for m in self.cells():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


class WideResNet_Open(nn.Cell):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet_Open, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, has_bias=False,pad_mode='pad')
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(alpha=0.1)
        # self.simclr_layer = nn.Sequential(
        #         nn.Linear(channels[3], 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 128),
        # )

        self.fc_mu = nn.Dense(channels[3], channels[3],has_bias=True)
        self.fc_logvar = nn.Dense(channels[3], channels[3],has_bias=True)

        self.fc1 = nn.Dense(channels[3], num_classes,has_bias=True)
        out_open = 2 * num_classes
        self.fc_open = nn.Dense(channels[3], out_open, has_bias=False)
        self.channels = channels[3]
        
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
        self.decoder = Data_Decoder_CIFAR(z_dim=channels[3])

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='leaky_relu'),
                    m.weight.shape, m.weight.dtype))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(initializer("ones", m.gamma.shape, m.gamma.dtype))
                m.beta.set_data(initializer("zeros", m.beta.shape, m.beta.dtype))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(initializer(
                    ms.common.initializer.XavierNormal(),
                    m.weight.shape, m.weight.dtype))
                if m.bias is not None:
                    m.bias.set_data(initializer("zeros", m.bias.shape, m.bias.dtype))

    def construct(self, x, feature=True, stats=False):
        #self.weight_norm()
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d(out, 1)
        out = out.view(-1, self.channels)
        mu, logvar = self.fc_mu(out), self.fc_logvar(out)
        out = self.reparameterize(mu, logvar)
        out_open = self.fc_open(out)
        rec_out = self.decoder(out)

        if stats:
            if feature:
                return self.fc1(out), out_open, out, rec_out, mu, logvar # self.simclr_layer(out)
            else:
                return self.fc1(out), out_open, rec_out, mu, logvar
        else:
            if feature:
                return self.fc1(out), out_open, out, rec_out # self.simclr_layer(out)
            else:
                return self.fc1(out), out_open, rec_out
        

    def reparameterize(self, mu, logvar):
        std = ops.exp(0.5 * logvar)
        eps = ops.UniformInt(std)
        return eps.mul(std).add_(mu)
    

    def weight_norm(self):
        w = self.fc_open.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc_open.weight.data = w.div(norm.expand_as(w))


    def disentangle(self, x, reverse=False):
        if reverse:
        # using gradiant reveral layer: use entropy minimization loss
        # else, use negative entropy
            x = self.grl(x)
        with ops.stop_gradient():
            x = self.fc1(x)
        return x

    


class ResBasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False,pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, has_bias=False,pad_mode='pad'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = ops.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = ops.ReLU(out)
        return out

#
class ResNet_Open(nn.Cell):
    def __init__(self, block, num_blocks, low_dim=128, num_classes=10):
        super(ResNet_Open, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Dense(512*block.expansion, low_dim)
        self.simclr_layer = nn.SequentialCell(
            nn.Dense(512*block.expansion, 128,has_bias=True),
            ops.ReLU(),
            nn.Dense(128, 128),
        )
        
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
        self.fc1 = nn.Dense(512*block.expansion, num_classes,has_bias=True)
        self.fc_open = nn.Dense(512*block.expansion, num_classes*2, has_bias=False)
        self.decoder = Data_Decoder_CIFAR(z_dim=512*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)

    def construct(self, x, feature=True):
        out = ops.ReLU(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = ops.AvgPool(out, 4)
        out = out.view(out.size(0), -1)
        out_open = self.fc_open(out)
        rec_out = self.decoder(out)

        if feature:
            return self.fc1(out), out_open, out, rec_out # self.simclr_layer(out)
        else:
            return self.fc1(out), out_open, rec_out

    def disentangle(self, x, reverse=True):
        with ops.stop_gradient():
            if reverse:
                x = self.grl(x)
            x = self.fc1(x)
        return x



def ResNet18(low_dim=128, num_classes=10):
   return ResNet_Open(ResBasicBlock, [2,2,2,2], low_dim, num_classes)



class FC(nn.Cell):
    def __init__(self, num_classes=6, z_dim=2, bias=True):
        super(FC, self).__init__()
        if not bias:
            self.fc = nn.Dense(z_dim, num_classes, has_bias=bias)
        else:
            self.fc = nn.Dense(z_dim, num_classes,has_bias=True)
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def construct(self, x, reverse=False):
        if reverse:
            x = self.grl(x)
        x = self.fc(x)
        return x

          
class Data_Decoder_CIFAR(nn.Cell):
    def __init__(self, hidden_dims = [256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Dense(z_dim, hidden_dims[0] * 4,has_bias=True)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.SequentialCell(
                    nn.Conv2dTranspose(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,pad_mode='pad'),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.SequentialCell(*modules)


        self.final_layer = nn.SequentialCell(
                            nn.Conv2dTranspose(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,pad_mode='pad'),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1,pad_mode='pad'),
                            nn.Sigmoid())


    def construct(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out



class Data_Decoder_MNIST(nn.Cell):

    def __init__(self, num_classes = 2, hidden_dims = [256, 128, 64, 32], z_dim = 1):
        super().__init__()
        self.decoder_input = nn.Dense(z_dim, hidden_dims[0] * 4)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.SequentialCell(
                    nn.Conv2dTranspose(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,pad_mode='pad'),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.SequentialCell(*modules)

        self.final_layer = nn.SequentialCell(
                            nn.Conv2dTranspose(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,pad_mode='pad'),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 4,pad_mode='pad'),
                            nn.Sigmoid())


    def construct(self, z_1, z_2):
        out = ops.concat((z_1, z_2), axis=1)
        out = self.decoder_input(out)
        out = out.view(-1, 256, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out





def build_wideresnet(depth, widen_factor, dropout, num_classes, open=True):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    build_func = WideResNet_Open if open else WideResNet
    return build_func(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes)
