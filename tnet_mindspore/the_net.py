from ResNet import resnet50,ResNet
import mindspore
import mindspore.nn as nn
from mindspore import nn, dataset, context
from mindspore import ops
from mindspore.common.initializer import Normal, initializer
from utils import z_repeat
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


pretrainedmodel = {
    "resnet50": "./checkpoints/resnet50_224_new.ckpt"
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)

class Interpolate(nn.Cell):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interpolate = ops.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
    def construct(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=True)
        return x


class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3)
        self.sigmoid = nn.Sigmoid()
    def construct(self, ftr):
        op_avg = ops.ReduceMean(keep_dims=True)
        # argmax = ops.ArgMaxWithValue()
        op_max = ops.ArgMaxWithValue(keep_dims=True,axis=1)
        # ftr_avg = ops.ReduceMean(ftr, keep_dims=True)
        ftr_avg=op_avg(ftr,1)
        # ftr_max, _ = ops.Maximum(ftr, dim=1,keep_dims=True)
        index, ftr_max = op_max(ftr)
        # ftr_max=op_max(ftr)
        ftr_cat = ops.concat([ftr_avg, ftr_max], axis=1)
        f_c=self.conv(ftr_cat)

        att_map = self.sigmoid(f_c)
        return att_map


def convblock(in_, out_, ks, st, pad):
    return nn.SequentialCell(
        nn.Conv2d(in_, out_, ks, st, pad_mode='pad',padding=pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )


class Decoder(nn.Cell):
    def __init__(self, in_1, in_2):
        super(Decoder, self).__init__()
        self.conv1 = convblock(in_1, 128, 3, 1, 1)
        self.conv_out = convblock(128, in_2, 3, 1, 1)

    def construct(self, pre,cur):
        cur_size = cur.shape[2:]
        pre = self.conv1(ops.interpolate(pre, sizes=cur_size, mode='bilinear',  coordinate_transformation_mode="align_corners"))
        fus = pre
        return self.conv_out(fus)

class CA(nn.Cell):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        # self.max_weight = nn.AdaptiveMaxPool2d(output_size=1)
        self.fus = nn.SequentialCell(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, padding=0),
        )
        self.c_mask = nn.Sigmoid()
    def construct(self, x):
        avg_map_c = self.avg_weight(x)

        fus_m=self.fus(avg_map_c)

        c_mask=self.c_mask(fus_m)
        return ops.mul(x, c_mask)

class FinalOut(nn.Cell):
    def __init__(self):
        super(FinalOut, self).__init__()
        self.ca =CA(128)
        self.score = nn.Conv2d(128, 1, 1, 1, pad_mode='same',padding=0)
    def construct(self,f1,f2,xsize):
        f1 = ops.concat((f1,f2),axis=1)
        f1 = self.ca(f1)
        score = ops.interpolate(self.score(f1), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")
        return score

class SaliencyNet(nn.Cell):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.c4=nn.Conv2d(2048*2,2048,kernel_size=1)
        self.c3=nn.Conv2d(1024*2, 1024, kernel_size=1)
        self.c2=nn.Conv2d(512*2,512,kernel_size=1)
        self.c1 = nn.Conv2d(256*2, 256, kernel_size=1)
        self.spa = SpatialAttention()
        self.ca4 = CA(2048*2)
        self.ca3 = CA(2048)
        self.ca2 = CA(1024)
        self.ca1 = CA(512)
        self.ca = CA(128)

        self.d4_r = Decoder(2048,1024)
        self.d3_r= Decoder(1024,512)
        self.d2_r= Decoder(512,256)
        self.d1_r = Decoder(256, 64)

        self.score = nn.Conv2d(128, 1, 1, 1, pad_mode='same',padding=0)
        self.score4 = nn.Conv2d(1024, 1, 1, 1, pad_mode='same',padding=0)
        self.score3 = nn.Conv2d(512, 1, 1, 1, pad_mode='same',padding=0)
        self.score2 = nn.Conv2d(256, 1, 1, 1, pad_mode='same',padding=0)
        self.score1 = nn.Conv2d(64, 1, 1, 1, pad_mode='same',padding=0)

    def construct(self,tt,r,r1,r2,r3,r4,t4,alpha,t,t1,t2,t3,aaa):
        xsize=tt.shape[2:]
        alpha5=z_repeat(alpha,int(aaa/32),int(aaa/32))
        tt4 = ops.mul(t4, 1 - alpha5)
        sp5 = self.spa(tt4)
        temp=ops.mul(r4,sp5)
        r4 = r4 + temp
        d4 = ops.mul(r4,alpha5)+ops.mul(t4,1-alpha5)
        d4 = ops.concat((d4, r4), axis=1)
        d4=self.ca4(d4)
        d4 = self.c4(d4)
        d4=self.d4_r(d4,r3)
        u4=d4

        alpha4 = z_repeat(alpha, int(aaa / 16), int(aaa / 16))
        tt3 = ops.mul(t3, 1 - alpha4)
        sp4 = self.spa(tt3)
        temp = ops.mul(d4, sp4)
        d4=d4+temp

        d3 = ops.mul(r3, alpha4) + ops.mul(t3, 1 - alpha4)
        d3 = ops.concat((d4, d3), axis=1)
        d3=self.ca3(d3)
        d3=self.c3(d3)
        d3 = self.d3_r(d3, r2)
        u3=d3

        alpha3 = z_repeat(alpha, int(aaa / 8), int(aaa / 8))
        tt2 = ops.mul(t2, 1 - alpha3)
        sp3 = self.spa(tt2)
        temp = ops.mul(d3, sp3)
        d3 = d3 + temp

        d2 = ops.mul(r2, alpha3) + ops.mul(t2, 1 - alpha3)
        d2 = ops.concat((d3, d2), axis=1)
        d2=self.ca2(d2)
        d2 = self.c2(d2)
        d2 = self.d2_r(d2, r1)
        u2=d2

        alpha2 = z_repeat(alpha, int(aaa / 4), int(aaa / 4))
        tt1 = ops.mul(t1, 1 - alpha2)
        sp2 = self.spa(tt1)
        temp = ops.mul(d2, sp2)
        d2 = d2 + temp

        d1 = ops.mul(r1, alpha2) + ops.mul(t1, 1 - alpha2)
        d1 = ops.concat((d2, d1), axis=1)
        d1=self.ca1(d1)
        d1 = self.c1(d1)
        d1 = self.d1_r(d1, r)
        u1=d1

        alpha1 = z_repeat(alpha, int(aaa / 4), int(aaa / 4))
        tt = ops.mul(t, 1 - alpha1)
        sp1 = self.spa(tt)
        temp = ops.mul(d1, sp1)
        d1 = d1 + temp

        d = ops.mul(r, alpha1) + ops.mul(t, 1 - alpha1)
        d = ops.concat((d, d1), axis=1)
        d=self.ca(d)
        result = ops.interpolate(self.score(d), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")
        u4=ops.interpolate(self.score4(u4), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")
        u3 = ops.interpolate(self.score3(u3), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")
        u2 = ops.interpolate(self.score2(u2), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")
        u1 = ops.interpolate(self.score1(u1), sizes=xsize, mode='bilinear', coordinate_transformation_mode="align_corners")

        return result,u4,u3,u2,u1

#baseline
class Baseline(nn.Cell):
    def __init__(self,channel=32):
        super(Baseline, self).__init__()

        #Backbone model
        self.resnet = ResNet('rgb')
        self.resnet_t = ResNet('rgbt')
        self.s_net = SaliencyNet()

        self.cc = nn.Conv2d(2048, 1, kernel_size=1)
        self.cc1 = nn.Conv2dTranspose(1, 1, kernel_size=16, pad_mode='pad',padding=4, stride=8)
        self.cc2 = nn.Conv2dTranspose(1, 1, kernel_size=16, pad_mode='pad',padding=4, stride=8)
        self.cc3 = nn.Conv2dTranspose(1, 1, kernel_size=8, pad_mode='pad',padding=2, stride=4)
        self.cc4 = nn.Conv2dTranspose(1, 1, kernel_size=4, pad_mode='pad',padding=1, stride=2)
        self.chutu = nn.Conv2dTranspose(1, 1, kernel_size=64, pad_mode='pad',padding=16, stride=32)
        self.sigmoid = nn.Sigmoid()

        self.gap=nn.AdaptiveAvgPool2d((1, 1))
        self.flatten=nn.Flatten()
        self.fc=nn.Dense(3*1*1,1)

        self.initialize_weights()

    def construct(self,x,x_t, L):

        aaa=x.shape[-1]
        aaa=int(aaa)
        alpha=self.gap(L)
        alpha=self.flatten(alpha)
        alpha=self.fc(alpha)
        alpha=self.sigmoid(alpha)
        #RGB
        #conv1
        tt=x
        x = self.resnet.conv1(x)
        x = self.resnet.norm(x)
        x = self.resnet.relu(x)
        x = self.resnet.max_pool(x)

        #conv2
        x1 = self.resnet.layer1(x)

        #conv3
        x2 = self.resnet.layer2(x1)

        #conv4
        x3 = self.resnet.layer3(x2)

        #conv5
        x4 = self.resnet.layer4(x3)

        #T
        x_t = self.resnet_t.conv1(x_t)
        x_t = self.resnet_t.norm(x_t)
        x_t = self.resnet_t.relu(x_t)
        x_t = self.resnet_t.max_pool(x_t)

        ttt = self.cc(x4)

        tt1 = self.cc1(ttt)

        tt1 = self.sigmoid(tt1)

        alpha1=z_repeat(alpha,int(aaa/4),int(aaa/4))

        tt1=ops.mul(tt1,alpha1)
        temp=ops.mul(x_t,tt1)
        x_t = x_t+temp

        x_t1 = self.resnet_t.layer1(x_t)
        tt2 = self.cc2(ttt)
        tt2 = self.sigmoid(tt2)
        alpha2 = z_repeat(alpha, int(aaa / 4), int(aaa / 4))
        tt2 = ops.mul(tt2, alpha2)
        temp = ops.mul(x_t1, tt2)
        x_t1 = x_t1 + temp

        x_t2 = self.resnet_t.layer2(x_t1)
        tt3 = self.cc3(ttt)
        tt3 = self.sigmoid(tt3)
        alpha3 = z_repeat(alpha, int(aaa / 8), int(aaa / 8))
        tt3 = ops.mul(tt3, alpha3)
        temp = ops.mul(x_t2, tt3)
        x_t2 = x_t2 + temp

        x_t3 = self.resnet_t.layer3(x_t2)
        tt4 = self.cc4(ttt)
        tt4 = self.sigmoid(tt4)
        alpha4 = z_repeat(alpha, int(aaa / 16), int(aaa / 16))
        tt4 = ops.mul(tt4, alpha4)
        temp = ops.mul(x_t3, tt4)
        x_t3 = x_t3 + temp

        x_t4 = self.resnet_t.layer4(x_t3)
        tt5 = self.sigmoid(ttt)
        alpha5 = z_repeat(alpha, int(aaa / 32), int(aaa / 32))
        tt5 = ops.mul(tt5, alpha5)
        temp = ops.mul(x_t4, tt5)
        x_t4 = x_t4 + temp

        #Decoder
        result_r,u4,u3,u2,u1 = self.s_net(tt,x,x1,x2,x3,x4, x_t4,alpha,x_t,x_t1,x_t2,x_t3,aaa)
        result_r=self.sigmoid(result_r)
        u4=self.sigmoid(u4)
        u3 = self.sigmoid(u3)
        u2 = self.sigmoid(u2)
        u1 = self.sigmoid(u1)
        ttt = self.chutu(ttt)
        ttt = self.sigmoid(ttt)
        return result_r,ttt,u4,u3,u2,u1

        #initialize the weights
    def initialize_weights(self):
        res50 = resnet50(pretrained=True)
        # pretrained_dict = res50.state_dict()
        pretrained_dict = res50.parameters_dict()
        all_params = {}
        for k, v in self.resnet.parameters_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif 'resnet' in k:
                name=k.split('resnet.')[1]
                v=pretrained_dict[name]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.parameters_dict().keys())
        load_param_into_net(self.resnet,all_params)

        all_params = {}
        for k, v in self.resnet_t.parameters_dict().items():
            if k=='resnet_t.conv1.weight':
                all_params[k]=v
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif 'resnet_t' in k:
                name = k.split('resnet_t.')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_t.parameters_dict().keys())
        load_param_into_net(self.resnet_t,all_params)

if __name__ == '__main__':
    import numpy as np
    import mindspore
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.random.randn(2, 3, 256, 256)
    x = mindspore.Tensor(x, mindspore.float32)
    y = np.random.randn(2, 1, 256, 256)
    y = mindspore.Tensor(y, mindspore.float32)
    z = np.random.randn(2, 3, 256, 256)
    z = mindspore.Tensor(z, mindspore.float32)
    model = Baseline()
    s1, ttt, u4, u3, u2, u1 = model(x, y,z)
    print(s1)
    print(s1.shape)
    print(u1.shape)


