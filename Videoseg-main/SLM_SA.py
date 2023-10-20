import os
import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.context as context

os.environ["CUDA_VISIBLE_DEVICES"]="0"
context.set_context(device_target="GPU")
bn_mom = 0.9

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, pad_mode='pad')


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Cell):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, padding=0, pad_mode='pad')
        self.bn1 = nn.BatchNorm3d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn2 = nn.BatchNorm3d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, padding=0, pad_mode='pad')
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(nn.Cell):
    def __init__(self, inplanes, interplanes, outplanes):
        super(segmenthead, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv3d(inplanes, interplanes, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.bn2 = nn.BatchNorm3d(interplanes, momentum=bn_mom)
        self.conv2 = nn.Conv3dTranspose(interplanes, interplanes, kernel_size=3, padding=1, stride=2, output_padding=1, has_bias=True, pad_mode='pad')
        self.bn3 = nn.BatchNorm3d(interplanes, momentum=bn_mom)
        self.conv3 = nn.Conv3d(interplanes, interplanes, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.bn4 = nn.BatchNorm3d(interplanes, momentum=bn_mom)
        self.conv4 = nn.Conv3dTranspose(interplanes, interplanes, kernel_size=3, padding=1, stride=2, output_padding=1, has_bias=True, pad_mode='pad')
        self.bn5 = nn.BatchNorm3d(interplanes, momentum=bn_mom)
        self.conv5 = nn.Conv3d(interplanes, interplanes//2, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.bn6 = nn.BatchNorm3d(interplanes//2, momentum=bn_mom)
        self.conv6 = nn.Conv3d(interplanes//2, outplanes, kernel_size=3, padding=1, has_bias=True, pad_mode='pad')
        self.relu = nn.ReLU()

    def construct(self, x):        
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        x = self.conv3(self.relu(self.bn3(x)))
        x = self.conv4(self.relu(self.bn4(x)))
        x = self.conv5(self.relu(self.bn5(x)))
        x = self.conv6(self.relu(self.bn6(x)))

        return x

class Selfattention(nn.Cell):
    """Self-attention with conv_qkv and reattention function"""
    
    def __init__(self, dim_in, dim_out, num_heads, patch_size=2, attn_drop=1.0, proj_drop=1.0, init_values=1e-4,
                 kernel_size=3, stride=1, padding=1, conv_qkv=True, layerscale=True,
                 reattention_heads=8, reattention_kernel_size=3, reattention=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = dim_out ** -0.5
        self.conv_qkv = conv_qkv
        self.reattention = reattention
        self.layerscale = layerscale
        self.softmax = nn.Softmax(axis=-1)
        
        if self.conv_qkv:
            self.proj_q = nn.SequentialCell([
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, pad_mode='pad', group=dim_in),
                nn.BatchNorm3d(dim_out, momentum=bn_mom),
                nn.ReLU()])
            self.proj_k = nn.SequentialCell([
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, pad_mode='pad', group=dim_in),
                nn.BatchNorm3d(dim_out, momentum=bn_mom),
                nn.ReLU()])
            self.proj_v = nn.SequentialCell([
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, pad_mode='pad', group=dim_in),
                nn.BatchNorm3d(dim_out, momentum=bn_mom),
                nn.ReLU()])
        else:
            self.proj_q = nn.Dense(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, has_bias=False)
            self.proj_k = nn.Dense(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, has_bias=False)
            self.proj_v = nn.Dense(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, has_bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if self.reattention:
            self.reattention_expansion = nn.Dense(num_heads,reattention_heads,has_bias=False)
            self.reattention_conv = nn.Conv2d(reattention_heads, reattention_heads, 
                                              kernel_size=reattention_kernel_size, 
                                              padding=reattention_kernel_size//2,stride=1,
                                              pad_mode='pad',group=reattention_heads) # depthwise-conv
            self.reattention_reduction = nn.Dense(reattention_heads,num_heads,has_bias=False)
        
        if self.layerscale:
            self.LayerScale = mindspore.Parameter(init_values*P.ones((dim_out), mindspore.float32), requires_grad=True)

    def Patch(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, self.num_heads, C//self.num_heads, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, D//self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 5, 7, 4, 6, 8, 2) # B*h*H'*W'*D'*k*k*k*d
        x = x.reshape(B, self.num_heads, H*W*D//(self.patch_size**3), self.patch_size**3*C//self.num_heads)
        return x
    
    def unPatch(self, x, B, C, H, W, D):
        x = x.reshape(B, self.num_heads, H//self.patch_size, W//self.patch_size, D//self.patch_size, self.patch_size, self.patch_size, self.patch_size, C//self.num_heads)
        x = x.permute(0, 1, 8, 2, 5, 3, 6, 4, 7) # B*h*d*H'*k*W'*k*D'*k
        x = x.reshape(B, C, H, W, D)
        return x
    
    def construct(self, x_q, x_k, x_v):
        B1, C1, H1, W1, D1 = x_q.shape
        B2, C2, H2, W2, D2 = x_k.shape
        
        if self.conv_qkv:
            q = self.proj_q(x_q)
            k = self.proj_k(x_k)
            v = self.proj_v(x_v) # B*C*H*W*D
            q = self.Patch(q) # B*H*N1*d
            k = self.Patch(k)
            v = self.Patch(v) # B*H*N2*d
        else:
            q = self.Patch(x_q) # B*H*N1*d
            k = self.Patch(x_k)
            v = self.Patch(x_v) # B*H*N2*d
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        attn_score = P.Einsum('bhlk,bhtk->bhlt')((q, k)) * self.scale # B*H*N1*N2
        attn = self.softmax(attn_score)
        if self.reattention:
            attn = self.reattention_expansion(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B*H*N1*N2
            attn = self.reattention_conv(attn) # B*H*N1*N2
            attn = self.reattention_reduction(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B*H*N1*N2  
        attn = self.attn_drop(attn)
        x = P.Einsum('bhlt,bhtv->bhlv')((attn, v)) # B*H*N1*d
        
        x = self.unPatch(x, B1, self.dim_out, H1, W1, D1) # B1*C1*H1*W1*D1
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.layerscale:
            x = x * self.LayerScale
        x = x.permute(0, 4, 1, 2, 3)
             
        return x
    
class FFN(nn.Cell):
    def __init__(self, d_in, d_hid, d_out=None, dropout=0.9):
        super().__init__()
        if d_out is None:
            d_out = d_hid
        self.w_1 = nn.Dense(d_in, d_hid)
        self.w_2 = nn.Dense(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = x.permute(0,2,3,4,1)
        
        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        
        x = x.permute(0,4,1,2,3)
        
        return x
    
class Norm(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(tuple([in_channels]))
        
    def construct(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
    
class TransForm_s(nn.Cell):
    def __init__(self, dim_in, dim_out, heads=4, patch_size=4,
                 conv_qkv=True, reattention=True, layerscale=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.SA = Selfattention(dim_in, dim_in, heads, patch_size=patch_size, 
                                conv_qkv=conv_qkv, reattention=reattention, layerscale=layerscale)
        self.FFN = FFN(dim_in, dim_out)
        self.Norm1 = Norm(dim_in)
        self.Norm2 = Norm(dim_out)
        if dim_out != dim_in:
            self.downconv = nn.SequentialCell([
                            nn.Conv3d(dim_in, dim_out, 1, has_bias=True),
                            nn.BatchNorm3d(dim_out, momentum=bn_mom),
                            nn.ReLU()])
        
    def construct(self, x_low, x_high):
        residual = x_high
        x_high = self.SA(x_high, x_low, x_low)
        x_high = self.Norm1(x_high)
        x_high += residual
        
        
        residual = self.downconv(x_high) if self.dim_out != self.dim_in else x_high
        x_high = self.FFN(x_high)
        x_high = self.Norm2(x_high)
        x_high += residual
        
        return x_high

class SLM_SA(nn.Cell):
    def __init__(self, block, layers, num_classes=2, dim_low=64, dim_high=32, dim_head=32, augment=False):
        super(SLM_SA, self).__init__()
        
        if block == 'basic':
            self.block = BasicBlock
        elif block == 'bottleneck':
            self.block = Bottleneck
        else:
            raise(Exception('The kind of block is invalid !'))
        self.augment = augment
        self.conv1 =  nn.SequentialCell([
                          nn.Conv3d(1,dim_high,kernel_size=3, stride=2, padding=1, has_bias=True, pad_mode='pad'),
                          nn.BatchNorm3d(dim_high, momentum=bn_mom),
                          nn.ReLU(),
                          nn.Conv3d(dim_high,dim_high,kernel_size=3, padding=1, has_bias=True, pad_mode='pad'),
                          nn.BatchNorm3d(dim_high, momentum=bn_mom),
                          nn.ReLU(),
                          nn.Conv3d(dim_high,dim_high,kernel_size=3, stride=2, padding=1, has_bias=True, pad_mode='pad'),
                          nn.BatchNorm3d(dim_high, momentum=bn_mom),
                          nn.ReLU(),
                          nn.Conv3d(dim_high,dim_high,kernel_size=3, padding=1, has_bias=True, pad_mode='pad'),
                          nn.BatchNorm3d(dim_high, momentum=bn_mom),
                          nn.ReLU()])
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(dim_high, dim_high, layers[0])
        self.layer2 = self._make_layer(dim_high, dim_high, layers[1])
        self.layer3 = self._make_layer(dim_high, dim_high, layers[2])
        self.layer4 = self._make_layer(dim_high, dim_high, layers[3])
        self.down1 = nn.SequentialCell([nn.Conv3d(dim_high, dim_low, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
                                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                                        nn.ReLU()])
        self.down2 = nn.SequentialCell([nn.Conv3d(dim_high, dim_low, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
                                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                                        nn.ReLU()])
        self.down3 = nn.SequentialCell([nn.Conv3d(dim_high, dim_low, kernel_size=3, stride=2, padding=1, pad_mode='pad'),
                                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                                        nn.ReLU()])           
        self.Transformer1 = TransForm_s(dim_low, dim_low)
        self.Transformer2 = TransForm_s(dim_low, dim_low)
        self.Transformer3 = TransForm_s(dim_low, dim_low)
        self.Transformero = TransForm_s(dim_high*3, dim_high)
        self.upconv1 = nn.SequentialCell([
                        nn.Conv3d(dim_high, dim_low, 1, has_bias=True),
                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                        nn.ReLU()])
        self.upconv2 = nn.SequentialCell([
                        nn.Conv3d(dim_high, dim_low, 1, has_bias=True),
                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                        nn.ReLU()])
        self.upconv3 = nn.SequentialCell([
                        nn.Conv3d(dim_high, dim_low, 1, has_bias=True),
                        nn.BatchNorm3d(dim_low, momentum=bn_mom),
                        nn.ReLU()])
        self.downconv1 = nn.SequentialCell([
                        nn.Conv3d(dim_low, dim_high, 1, has_bias=True),
                        nn.BatchNorm3d(dim_high, momentum=bn_mom),
                        nn.ReLU()])
        self.downconv2 = nn.SequentialCell([
                        nn.Conv3d(dim_low, dim_high, 1, has_bias=True),
                        nn.BatchNorm3d(dim_high, momentum=bn_mom),
                        nn.ReLU()])
        self.downconv3 = nn.SequentialCell([
                        nn.Conv3d(dim_low, dim_high, 1, has_bias=True),
                        nn.BatchNorm3d(dim_high, momentum=bn_mom),
                        nn.ReLU()])
        if self.augment:
            self.seghead_extra = segmenthead(dim_high, dim_head, num_classes)            
        self.seghead = segmenthead(dim_high, dim_head, num_classes)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * self.block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv3d(inplanes, planes * self.block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes * self.block.expansion, momentum=bn_mom),
            ])

        layers = []
        layers.append(self.block(inplanes, planes, stride, downsample))
        inplanes = planes * self.block.expansion
        for i in range(1, blocks):
            layers.append(self.block(inplanes, planes, stride=1, no_relu=False))

        return nn.SequentialCell(layers)

    def construct(self, x):
        layers = []
        skips = []
        x = self.conv1(x)
        
        x = self.layer1(x)
        layers.append(x)
        x_ = self.down1(layers[0])
        
        x = self.layer2(x)
        layers.append(x)
        x_ = self.downconv1(self.Transformer1(x_, self.upconv1(layers[1])))
        skips.append(x_)
        if self.augment:
            temp = x_
        x = x + x_
        x_ = self.down2(x_)
        
        x = self.layer3(x)
        layers.append(x)
        x_ = self.downconv2(self.Transformer2(x_, self.upconv2(layers[2])))
        skips.append(x_)
        x = x + x_
        x_ = self.down3(x_)
        
        x = self.layer4(x)
        layers.append(x)
        x_ = self.downconv3(self.Transformer3(x_, self.upconv3(layers[3])))
        skips.append(x_)

        cat_op = P.Concat(1)
        out_fea = cat_op(skips)
        out = self.Transformero(out_fea, out_fea) 
        out = self.seghead(out)

        if self.augment: 
            out_extra = self.seghead_extra(temp)
            return [out, out_extra]
        else:
            return out    
    
    
if __name__ == '__main__':
    model = SLM_SA('basic', [2, 2, 2, 2, 2])
    x = P.UniformReal(seed=2)((1,1,32,32,32))
    print(model(x).shape)

    

