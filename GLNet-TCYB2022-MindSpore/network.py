from modules import *
# set_context(mode=PYNATIVE_MODE)
ms.set_context(mode=ms.PYNATIVE_MODE)
class GLNet(nn.Cell):
    def __init__(self, return_loss=True):
        super(GLNet, self).__init__()
        self.return_loss = return_loss
        self.M = 5
        self.D = 512
        self.backbone = BackboneBuilder()
        self.SpatialAttention1 = SpatialAttention()
        self.ChannelAttention1 = ChannelAttention(self.D)
        self.SpatialAttention2 = SpatialAttention()
        self.ChannelAttention2 = ChannelAttention(self.D)
        self.SpatialAttention3 = SpatialAttention()
        self.ChannelAttention3 = ChannelAttention(self.D)
        self.co_att = co_att(self.D)
        self.cov3d2 = nn.SequentialCell(nn.Conv3d(self.D, self.D, kernel_size=(3, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d1 = nn.SequentialCell( nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1),has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d3 = nn.SequentialCell( nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1),has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d4 = nn.SequentialCell(nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d5 = nn.SequentialCell(nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d6 = nn.SequentialCell(nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.cov3d7 = nn.SequentialCell(nn.Conv3d(self.D, self.D, kernel_size=(2, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), has_bias=True, pad_mode="pad"),
                                   nn.ReLU())
        self.aewf = AEWF(self.D)
        self.de_1 = DE(self.D)
        self.de_2 = DE(self.D//2)
        self.de_3 = DE(self.D//4)
        self.cosal_head = CosalHead(self.D//8)
        self.reshape = P.Reshape()
        self.cat = P.Concat(axis=2)
        self.cat2 = P.Concat(axis=1)
        self.unsqueeze = ops.ExpandDims()
    def construct(self, grp_images):
        # print("grp_images====================",grp_images.shape)   
        M = self.M
        D = self.D
        height, width = 160, 160
        # print(grp_images.shape)
        Bg = grp_images.shape[0]
        raw_ftr = self.backbone(self.reshape(grp_images,(Bg * M, 3, height, width)))
        _, D, H, W = raw_ftr.shape  
        UF = self.reshape(raw_ftr,(Bg, M, D, H, W))
        UF_bag = []
        for m in range(M):
            UF_m = UF[:, m, :, :, :]  
            UF_bag.append(self.unsqueeze(UF_m,2))
        CUF = self.cat(UF_bag)
        CUF = self.cov3d3(self.cov3d2(self.cov3d1(CUF))).squeeze(2)
        CUF = self.ChannelAttention1(CUF)
        CUF = self.SpatialAttention1(CUF)
        #pairwise
        X_bag = []
        for m in range(M):
            PF_bag = []
            for n in range(M):
                if m!=n:
                    PFm = self.co_att(UF_bag[m].squeeze(2), UF_bag[n].squeeze(2))
                    PFm = self.ChannelAttention2(PFm)
                    PFm = self.SpatialAttention2(PFm)
                    PF_bag.append(self.unsqueeze(PFm,2)) 
                   
            PUF01 = self.cov3d4(self.cat((PF_bag[0], PF_bag[1])))  
            PUF12 = self.cov3d5(self.cat((PUF01, PF_bag[2])))  
            PUF23 = self.cov3d6(self.cat((PUF12, PF_bag[3]))) 
            CUF_PUF = self.cov3d7(self.cat((self.unsqueeze(CUF,2), PUF23))).squeeze(2)
            CUF_PUF = self.ChannelAttention3(CUF_PUF)
            CUF_PUF = self.SpatialAttention3(CUF_PUF)          
            UF_m = UF[:, m, :, :, :]  
            co_saliency_features = self.aewf(CUF_PUF, UF_m)  
            X_bag.append(self.unsqueeze(co_saliency_features,1))
        X = self.cat2(X_bag) 
        X_1 = self.de_1(X) 
        X_2 = self.de_2(X_1)
        X_3 = self.de_3(X_2) 
        cosod_maps = self.reshape(self.cosal_head(X_3.view(Bg * M, D // 8, height, width)),(Bg, M, 1, height,width))
        return cosod_maps

    
if __name__ == '__main__':
    ms.set_context(device_target="GPU")
    [3, 5, 3, 160, 160]
    x = ms.Tensor(np.ones([3, 5, 3, 160, 160]).astype(np.float32))
    print("x====================",x.shape)   
    model = GLNet()
    s_global =  model(x)
    print(s_global.shape)
