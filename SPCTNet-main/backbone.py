from mindspore import ops, nn


class ResNeXtBottleneck(nn.Cell):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            pad_mode='pad',
            padding=1,
            group=cardinality,
            has_bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt3D(nn.Cell):

    def __init__(self, block, layers, cardinality=32):
        self.inplanes = 16
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=(2, 2, 1),pad_mode='pad', padding=1, has_bias=False)  # [32,96]
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 32, layers[0], cardinality,stride=2)  # [128,48]
        self.layer2 = self._make_layer(block, 64, layers[1], cardinality, stride=(2, 2, 2))  # [256,24]
        self.layer3 = self._make_layer(block, 128, layers[2], cardinality, stride=2)  # [512,12]
        self.layer4 = self._make_layer(block, 256, layers[3], cardinality, stride=(2, 2, 2))  # [1024,12]


    def _make_layer(self, block, planes, blocks, cardinality, stride=(1,1,1)):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        has_bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BackBone3D(nn.Cell):
    def __init__(self):
        super(BackBone3D, self).__init__()
        net = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3])
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        # and if we use the resnet3d-101, change the block list with [3, 4, 23, 3]
        net = list(net.children())
        self.layer0 = nn.SequentialCell(*net[:2])
        # the layer0 contains the first convolution, bn and relu
        self.layer1 = nn.SequentialCell(*net[2:4])
        # the layer1 contains the first pooling and the first 3 bottle blocks
        self.layer2 = net[4]
        # the layer2 contains the second 4 bottle blocks
        self.layer3 = net[5]
        # the layer3 contains the media bottle blocks
        # with 6 in 50-layers and 23 in 101-layers
        self.layer4 = net[6]

    def construct(self, x):
        layer0 = self.layer0(x)  # [32,96]
        layer1 = self.layer1(layer0)  # [128,48]
        layer2 = self.layer2(layer1)  # [256,24]
        layer3 = self.layer3(layer2)  # [512,12]
        layer4 = self.layer4(layer3)  # [1024,12]
        return layer4


if __name__ == '__main__':
    backbone = BackBone3D()
    x = ops.randn(1,1,192,192,48)
    y = backbone(x)
    print(y.shape)
