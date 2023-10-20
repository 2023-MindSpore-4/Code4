from utils import *

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


class CB1(nn.Cell):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB1, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, pad_mode='pad', kernel_size=1, has_bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def construct(self, x):
        # x: [B, in_channels, H, W]
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = self.relu(y)
        return y
    
class CB3(nn.Cell):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB3, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        # x: [B, in_channels, H, W]
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = self.relu(y)
        return y

def _bn1(channel):
    return nn.BatchNorm1d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

class FC(nn.Cell):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(FC, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu 
        self.linear = nn.Dense(in_channels, out_channels, has_bias=False, bias_init=0)
        if self.use_bn:
            self.bn = _bn1(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        y = self.linear(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = self.relu(y)
        return y
        
        
class ChannelAttention(nn.Cell):
    def __init__(self, in_channels, sqz_ratio=4):
        super(ChannelAttention, self).__init__()
        # self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = ops.ReduceMean(keep_dims=True)
        self.max_pooling = ops.ReduceMax(keep_dims=True)
        self.fc_1 = FC(in_channels, in_channels//sqz_ratio, True, True)
        self.fc_2 = FC(in_channels//sqz_ratio, in_channels, True, False)
        self.sigmoid =  nn.Sigmoid()
        self.unsqueeze = ops.ExpandDims()
    def construct(self, ftr):
        #print('ChannelAttention',ftr.shape)
        #print('avg_pooling',self.avg_pooling(ftr,(2, 3)).shape)
        avg_out = self.avg_pooling(ftr,(2, 3)).squeeze(-1).squeeze(-1)
        max_out = self.max_pooling(ftr,(2, 3)).squeeze(-1).squeeze(-1) 
        avg_weights = self.fc_2(self.fc_1(avg_out))
        max_weights = self.fc_2(self.fc_1(max_out))
        weights = self.sigmoid(avg_weights + max_weights) 
        return ftr * self.unsqueeze(self.unsqueeze(weights,-1),-1) + ftr

    
class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.fuse = CB3(2, 1, False, False)
        self.mean = P.ReduceMean(keep_dims=True)
        self.max = P.ReduceMax(keep_dims=True)
        self.concat = P.Concat(axis=1)
        self.sigmoid =  nn.Sigmoid()
    def construct(self, ftr):
        avg_out = self.mean(ftr, 1) 
        max_out = self.max(ftr, 1)
        cat_out = self.concat((avg_out, max_out)) 
        #print('cat_out==================',cat_out.shape)  
        cat_out = self.fuse(cat_out)
        #print('cat_out',cat_out.shape)
        sam = self.sigmoid(cat_out) 
        return sam*ftr + ftr

class BackboneBuilder(nn.Cell):
    def __init__(self, bb_load_path='./Checkpoints/warehouse/backbone.ckpt'):
        super(BackboneBuilder, self).__init__()
        dr = [1, 1/2, 1/4, 1/8, 1/8]
        ch = [64, 128, 256, 512, 512]
        # backbone  torch.load 需要修改
        ms_backbone = BackBone()
        ms.load_checkpoint(bb_load_path, ms_backbone)
        # self.bb = torch.load(bb_load_path)
        self.bb = ms_backbone
        self.SA_1 = SpatialAttention()
        self.SA_2 = SpatialAttention()
        self.SA_3 = SpatialAttention()
        self.CA_4 = ChannelAttention(ch[3])
        self.CA_5 = ChannelAttention(ch[4]) 
    def construct(self, Im):
        f1_1 = self.bb.C1(Im)
        #print('f1_1==================',f1_1.shape)    
        F1 = self.SA_1(self.bb.C1(Im))
        F2 = self.SA_2(self.bb.C2(F1)) 
        F3 = self.SA_3(self.bb.C3(F2))
        #print(' self.bb.C4(F3)==================', self.bb.C4(F3).shape)  
        #print(' self.bb.C4(F3)==================', self.bb.C4(F3).shape)    
        F4 = self.CA_4(self.bb.C4(F3)) 
        F5 = self.CA_5(self.bb.C5(F4))
        return F5

class co_att(nn.Cell):

    def __init__(self,in_dim):
        super(co_att,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim , in_dim , kernel_size= 1, has_bias=True)
        self.key_conv = nn.Conv2d(in_dim ,  in_dim , kernel_size= 1, has_bias=True)
        self.value_conv = nn.Conv2d(in_dim , in_dim , kernel_size= 1, has_bias=True)
        self.softmax  = P.Softmax(axis=-1)
        self.reshape = P.Reshape()
        self.tranpose = P.Transpose()
        self.batmatmul = P.BatchMatMul()
        self.max = P.ReduceMax(keep_dims=True)

    def construct(self,x,y):
        B, C, H, W = x.shape
        #print('co_att',x.shape)
        P = H * W
        xproj_query = self.query_conv(x)
        xproj_query = self.tranpose(self.reshape(xproj_query, (B, -1, P)), (0, 2, 1)) 
        #print('xproj_query',xproj_query.shape)
        yproj_key =  self.key_conv(y)
        yproj_key = self.reshape(yproj_key, (B, -1, P))
        #print('yproj_key',yproj_key.shape)
        energy = self.batmatmul(xproj_query, yproj_key)
        #print('energy',energy.shape)
        energy = self.max(energy, axis=2)
        #print('energy',energy.shape)
        # energy = torch.max(energy , dim=2, keepdim=True)[0]
        #print('self.reshape(self.softmax(energy), (B, 1, H, W))',self.softmax(energy).shape)
        #print(B, 1, H, W)
        attention = self.reshape(self.softmax(energy), (B, 1, H, W))
        # attention = self.softmax(energy).view(m_batchsize,1,width,height)
        xout = x * attention + x
        return xout
    
class AEWF(nn.Cell):
    def __init__(self, in_channels):
        super(AEWF, self).__init__()
        self.channel_reduction = CB1(in_channels*2, in_channels, True, True)
        self.importance_estimator = nn.SequentialCell(ChannelAttention(in_channels),
                                   CB3(in_channels, in_channels//2, True, True),
                                   CB3(in_channels//2, in_channels//2, True, True),
                                   CB3(in_channels//2, in_channels, True, False), nn.Sigmoid())
        self.cat = P.Concat(axis=1)
    def construct(self, group_semantics, individual_features):
        ftr_concat_reduc = self.channel_reduction(self.cat((group_semantics, individual_features)))
        P = self.importance_estimator(ftr_concat_reduc) 
        co_saliency_features = group_semantics * P + individual_features * (1-P) 
        return co_saliency_features  

    
class DE(nn.Cell):
    def __init__(self, in_channels):
        super(DE, self).__init__()
        self.channel_reduction = CB1(in_channels, in_channels//2, True, True)
        self.deconv = CB3(in_channels//2, in_channels//2, True, True)
        self.reshape = P.Reshape()
    def construct(self, X): 
        # [B, M, D, H, W] = X.size()
        B, M, D, H, W = X.shape
        X_US2 = self.deconv(US2(self.channel_reduction(self.reshape(X,(B*M, D, H, W))))) 
        return self.reshape(X_US2,(B, M, D//2, 2*H, 2*W))
    
    
class CosalHead(nn.Cell):
    def __init__(self, in_channels):
        super(CosalHead, self).__init__()
        self.output = nn.SequentialCell(CB3(in_channels, in_channels*4, True, True),
                           CB3(in_channels*4, in_channels*4, True, True),
                           CB3(in_channels*4, 1, False, False), nn.Sigmoid())
    def construct(self, x):
        return self.output(x)
        
    