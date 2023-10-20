from mindspore import nn, ops
from .backbone import BackBone3D
from .transformer import TransformerBlock


class AttentionBlock(nn.Cell):
    def __init__(self, in_channels, skip_channels, mid_channels):
        super(AttentionBlock, self).__init__()
        self.W_skip = nn.SequentialCell(nn.Conv3d(skip_channels, mid_channels, kernel_size=1, has_bias=False),
                                    nn.BatchNorm3d(mid_channels))
        self.W_x = nn.SequentialCell(nn.Conv3d(in_channels, mid_channels, kernel_size=1, has_bias=False),
                                 nn.BatchNorm3d(mid_channels))
        self.psi = nn.SequentialCell(nn.Conv3d(mid_channels, 1, kernel_size=1, has_bias=False),
                                 nn.BatchNorm3d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU()

    def construct(self, x_skip, x):
        x_skip = self.W_skip(x_skip)
        x = self.W_x(x)
        out = self.psi(self.relu(x_skip + x))
        return out


class SPCTNet(nn.Cell):
    def __init__(self,num_classes=2):
        super(SPCTNet, self).__init__()
        self.backbone = BackBone3D()

        self.down4 = nn.SequentialCell(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.down3 = nn.SequentialCell(
            nn.Conv3d(256, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.down2 = nn.SequentialCell(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.down1 = nn.SequentialCell(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.tran1 = TransformerBlock(in_channels=64,layers=2,mlp_dim=128,patch_size=4)
        self.tran2 = TransformerBlock(in_channels=64,layers=2,mlp_dim=128,patch_size=4)
        self.tran3 = TransformerBlock(in_channels=64,layers=2,mlp_dim=128,patch_size=2)
        self.tran4 = TransformerBlock(in_channels=64,layers=2,mlp_dim=128,patch_size=1)

        self.fuse1 = nn.SequentialCell(
            nn.Conv3d(256, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, pad_mode='pad', padding=1), nn.BatchNorm3d(64), nn.ReLU()
        )
        self.attention4 = AttentionBlock(64, 64, 64)
        self.attention3 = AttentionBlock(64, 64, 64)
        self.attention2 = AttentionBlock(64, 64, 64)
        self.attention1 = AttentionBlock(64, 64, 64)

        self.refine4 = nn.SequentialCell(
            nn.Conv3d(128, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            TransformerBlock(in_channels=64, layers=2, mlp_dim=128, patch_size=4)
        )
        self.refine3 = nn.SequentialCell(
            nn.Conv3d(128, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            TransformerBlock(in_channels=64, layers=2, mlp_dim=128, patch_size=4)
        )
        self.refine2 = nn.SequentialCell(
            nn.Conv3d(128, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            TransformerBlock(in_channels=64, layers=2, mlp_dim=128, patch_size=4)
        )
        self.refine1 = nn.SequentialCell(
            nn.Conv3d(128, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            TransformerBlock(in_channels=64, layers=2, mlp_dim=128, patch_size=4)
        )

        self.last_refine = nn.SequentialCell(
            nn.Conv3d(256, 64, kernel_size=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, pad_mode='pad', padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, pad_mode='pad', padding=1), nn.BatchNorm3d(64), nn.ReLU())

        self.predict4 = nn.Conv3d(64, num_classes, kernel_size=1)
        self.predict3 = nn.Conv3d(64, num_classes, kernel_size=1)
        self.predict2 = nn.Conv3d(64, num_classes, kernel_size=1)
        self.predict1 = nn.Conv3d(64, num_classes, kernel_size=1)

        self.predict = nn.Conv3d(64, num_classes, kernel_size=1)

    def construct(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        # FPN, Top-down
        down4 = self.down4(layer4)
        down3 = ops.add(
            ops.upsample(down4, size=layer3.size()[2:], mode='trilinear'),
            self.down3(layer3)
        )
        down2 = ops.add(
            ops.upsample(down3, size=layer2.size()[2:], mode='trilinear'),
            self.down2(layer2)
        )
        down1 = ops.add(
            ops.upsample(down2, size=layer1.size()[2:], mode='trilinear'),
            self.down1(layer1)
        )

        down4 = ops.upsample(self.tran4(down4), size=layer1.size()[2:], mode='trilinear')
        down3 = ops.upsample(self.tran3(down3), size=layer1.size()[2:], mode='trilinear')
        down2 = ops.upsample(self.tran2(down2), size=layer1.size()[2:], mode='trilinear')
        down1 = self.tran1(down1)

        fuse1 = self.fuse1(ops.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(fuse1, down4)
        attention3 = self.attention3(fuse1, down3)
        attention2 = self.attention2(fuse1, down2)
        attention1 = self.attention1(fuse1, down1)

        # 把上一步获得的 attention maps 应用到不同尺度的 features maps 上
        refine4 = self.refine4(ops.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(ops.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(ops.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(ops.cat((down1, attention1 * fuse1), 1))

        refine = self.last_refine(ops.cat((refine1, refine2, refine3, refine4), 1))

        predict4 = self.predict4(refine4)
        predict3 = self.predict3(refine3)
        predict2 = self.predict2(refine2)
        predict1 = self.predict1(refine1)

        predict = self.predict(refine)
        # print(predict.shape)

        predict1 = ops.upsample(predict1, size=x.size()[2:], mode='trilinear')
        predict2 = ops.upsample(predict2, size=x.size()[2:], mode='trilinear')
        predict3 = ops.upsample(predict3, size=x.size()[2:], mode='trilinear')
        predict4 = ops.upsample(predict4, size=x.size()[2:], mode='trilinear')

        predict = ops.upsample(predict, size=x.size()[2:], mode='trilinear')

        if self.training:
            return predict1, predict2, predict3, predict4, predict
        else:
            return predict


if __name__ == '__main__':
    x = ops.randn(1,1,192,192,48)
    net = SPCTNet()
    # net.eval()
    y = net(x)
    for p in y:
        print(p.shape)