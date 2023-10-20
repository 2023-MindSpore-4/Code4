from functools import partial
import mindspore as ms
from mindspore import dtype as mstype
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision as transforms
from mindspore import nn, ops
from typing import Optional, Dict
from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer
from mindspore import Parameter
from mindspore.ops import functional as F


from ipdb import set_trace as stxx

__all__ = [
'deit_base_patch16_224',
'student_deit_base_patch16_224_6layer'
]

class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches =  (img_size // patch_size) ** 2
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape        
        if self.flatten:

            x = ops.reshape(x, (B,C, H * W))
            x = ops.transpose(x, (0, 2, 1))
        x = self.norm(x)
        return x

class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p = drop)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(p = drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = ops.meshgrid(ops.arange(W).to(x.device), ops.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = ops.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = ops.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = ops.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = ops.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else ops.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=mstype.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.

#     """

#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     if keep_prob > 0.0 and scale_by_keep:
#         random_tensor.div_(keep_prob)
#     return x * random_tensor

# class DropPath(nn.Cell):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None, scale_by_keep=True):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#         self.scale_by_keep = scale_by_keep

#     def construct(self, x):
#         return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0) # always be 0

        self.shape = ops.Shape()
        self.ones = ops.Ones()
        self.dropout = nn.Dropout(p=1 - float(self.keep_prob.value()))

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x) # B N C
            mask = self.ones((x_shape[0], 1, 1), ms.float32)
            x = self.dropout(mask)*x
        return x
    

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8,qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(p = attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p = proj_drop)
        
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B, N, C = x.shape
        
        qkv = ops.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = self.q_matmul_k(q, k) 
        attn = ops.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, N, C))
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn
    
# class Attention(nn.Cell):
#     def __init__(self,
#                  dim: int,
#                  num_heads: int = 8,
#                  keep_prob: float = 1.0,
#                  attention_keep_prob: float = 1.0):
#         super(Attention, self).__init__()

#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = ms.Tensor(head_dim ** -0.5)

#         self.qkv = nn.Dense(dim, dim * 3)
#         self.attn_drop = nn.Dropout(1.0-attention_keep_prob)
#         self.out = nn.Dense(dim, dim)
#         self.out_drop = nn.Dropout(1.0-keep_prob)
#         self.attn_matmul_v = ops.BatchMatMul()
#         self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
#         self.softmax = nn.Softmax(axis=-1)

#     def construct(self, x):
#         """Attention construct."""
#         b, n, c = x.shape
#         qkv = self.qkv(x)
#         qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
#         qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
#         q, k, v = ops.unstack(qkv, axis=0)
#         attn = self.q_matmul_k(q, k)
#         attn = ops.mul(attn, self.scale)
#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)
#         out = self.attn_matmul_v(attn, v)
#         out = ops.transpose(out, (0, 2, 1, 3))
#         out = ops.reshape(out, (b, n, c))
#         out = self.out(out)
#         out = self.out_drop(out)

#         return out

class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):

        x_res = x

        x = self.norm1(x)
        x, attn = self.attn(x)
        x = x_res + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class VisionTransformer(nn.Cell):
    """ Vision Transformer

    A Mindspore impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Cell): patch embedding layer
            norm_layer: (nn.Cell): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(ops.zeros((1, 1, embed_dim)))
        self.dist_token = Parameter(ops.zeros((1, 1, embed_dim))) if distilled else None
        self.pos_embed = Parameter(ops.zeros((1, num_patches + self.num_tokens, embed_dim)))
        self.pos_drop = nn.Dropout(p = drop_rate)
        self.depth = depth
        dpr = [x.value() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.SequentialCell(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


        self.norm = norm_layer((embed_dim,))

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.SequentialCell(OrderedDict([
                ('fc', nn.Dense(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)        
        cls_token =  ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
        if self.dist_token is None:
            x = ops.cat((cls_token, x), axis=1)
        else:
            x = ops.cat((cls_token, ops.tile(self.dist_token(x.dtype),(x.shape[0], -1, -1)), x), axis=1)
        x = self.pos_drop(x + self.pos_embed)
        
        # x = self.blocks(x)
        hiddenlayer_trans = []
        hiddenlayer_attn = []

        for i in range(self.depth):
            # x, attn = eval('self.trans_block_' + str(i))(x)
            x, attn = self.blocks[i](x)
            hiddenlayer_trans.append(x)
            hiddenlayer_attn.append(attn)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), hiddenlayer_trans, hiddenlayer_attn
        else:
            return x[:, 0], x[:, 1]

    def construct(self, x):
        x, hiddenlayer_trans, hiddenlayer_attn = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training: # and not .jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, hiddenlayer_trans, hiddenlayer_attn


class StudentVisionTransformer(nn.Cell):
    """ Vision Transformer

    A Mindspore impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Cell): patch embedding layer
            norm_layer: (nn.Cell): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(ops.zeros((1, 1, embed_dim)))
        self.dist_token = Parameter(ops.zeros((1, 1, embed_dim))) if distilled else None
        self.pos_embed = Parameter(ops.zeros((1, num_patches + self.num_tokens, embed_dim)))
        self.pos_drop = nn.Dropout(p = drop_rate)
        self.depth = depth

        dpr = [x.value() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.SequentialCell(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])


        self.norm = norm_layer((embed_dim,))

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.SequentialCell(OrderedDict([
                ('fc', nn.Dense(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


        # prototype
        self.num_cluster = 24 * 3
        self.centroids3 = Parameter(ops.rand(self.num_cluster, self.embed_dim))

        self.netvlad3 = nn.SequentialCell(
        nn.Conv2d(self.embed_dim, self.num_cluster, 1)
        )
        self.cat_transform3 = nn.SequentialCell(
            nn.Dense(self.embed_dim * 3, self.embed_dim),
            nn.ReLU(),
            # nn.Conv2d(1024, 1024, kernel_size=1),
            # nn.ReLU(inplace=False),
        )
        self.transform3 = nn.SequentialCell(
            nn.Conv2d(self.embed_dim*2, self.embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.ReLU(),
        )

        self.vladfc3 = nn.Dense(self.num_cluster * self.embed_dim, self.embed_dim)

        self.encode3 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.aug_transform3 = nn.SequentialCell(
            nn.Conv1d(self.embed_dim*2 , self.embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.ReLU(),
        )


    def vlad3(self, scene, centroid):
        scene = scene.permute(0, 2, 1).unsqueeze(-1).contiguous()
        x = scene
        N, C, W, H = x.shape[0:]

        x = F.normalize(x, p=2, dim=1)  # .Size([64, 384, 197, 1])
        soft_assign = self.netvlad3(x)  # .Size([64, 72, 197, 1])

        soft_assign = F.softmax(soft_assign, axis=1) # .Size([64, 72, 197, 1])
        soft_assign = soft_assign.reshape(soft_assign.shape[0], soft_assign.shape[1], -1) # .Size([64, 72, 197])

        x_flatten = x.reshape(N, C, -1)

        # centroid = self.alpha3 * center + self.beta3  # 仿射变换

        x1 = x_flatten.broadcast_to(self.num_cluster, -1, -1, -1).permute(1, 0, 2, 3) # .Size([64, 72, 384, 197])
        x2 = centroid.broadcast_to(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0) # .Size([1, 72, 384, 197])

        residual = x1 - x2 # .Size([64, 72, 384, 197])
        residual = residual * soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.reshape(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        vlad = self.vladfc3(vlad)  # .Size([64, 384])
        # 维度变换
        vlad = ops.repeat(vlad.unsqueeze(2).unsqueeze(3),(1, 1, W, H))  # .Size([64, 384, 197, 1])

        vlad = ops.cat((scene, vlad), axis=1)  # .Size([64, 768, 197, 1])
        vlad = self.transform3(vlad).squeeze(3).permute(0,2,1)# # .Size([64, 384, 197, 1]) -> .Size([64,197, 384])
        return vlad

    def aug3(self, xs, centroid):
        """
        :param xs: .Size([64, 197, 384])
        :param centroid: .Size([72, 1024])
        :return:.Size([64, 197, 384])
        """

        xs = xs.permute(0,2,1)#  # .Size([64, 384, 197])
        aug = self.encode3(xs.reshape(xs.shape[0], xs.shape[1], -1))  #.Size([64, 384, 197])  这个encode3 一定要一样么
        new_center = self.encode3(ops.tile(centroid.unsqueeze(0),(xs.shape[0], 1, 1)).permute(0, 2, 1)).permute(0, 2, 1) # .Size([64, 72, 384])

        align = F.softmax(ops.matmul(new_center, aug), axis=1) # .Size([64, 72, 197])

        aug_feature = ops.matmul(new_center.permute(0, 2, 1), align) # .Size([64, 384, 197])
        aug = ops.cat((aug, aug_feature), axis=1) # .Size([64, 768, 197])
        aug = ops.relu(self.aug_transform3(aug) + xs.reshape(xs.shape[0], xs.shape[1], -1)) # .Size([64, 384, 197])
        aug = aug.permute(0,2,1)#
        # aug = aug.reshape(*xs.size())
        return aug

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, new_teacher_convs, is_Train):
#         stxx()
        x = self.patch_embed(x)
        cls_token =  ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
        if self.dist_token is None:
            x = ops.concat((cls_token, x), axis=1)
        else:
            x = ops.concat((cls_token, ops.tile(self.dist_token(x.dtype),(x.shape[0], -1, -1)), x), axis=1)
        x = self.pos_drop(x + self.pos_embed)

        # x = self.blocks(x)
        hiddenlayer_trans = []
        hiddenlayer_attn = []

        for i in range(self.depth):
            x, attn = self.blocks[i](x)
            hiddenlayer_trans.append(x)
            hiddenlayer_attn.append(attn)

        # proto3
        if is_Train:
            proto_ipt = ops.cat((new_teacher_convs[0],new_teacher_convs[1],new_teacher_convs[2]), axis=-1)  # .Size([64, 2048, 7, 7])
            proto_ipt = self.cat_transform3(proto_ipt)  # .Size([64, 1024, 7, 7])
            vlad3 = self.vlad3(proto_ipt, self.centroids3)  # .Size([64, 1024, 7, 7])
            vlad3 = self.aug3(vlad3, self.centroids3)
        else:
            vlad3 = self.aug3(x, self.centroids3)

        vlad3 = self.norm(vlad3)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), self.pre_logits(vlad3[:, 0]), hiddenlayer_trans, hiddenlayer_attn
        else:
            return x[:, 0], x[:, 1]

    def construct(self, x, new_teacher_convs, is_Train):
        x, aug_x, hiddenlayer_trans, hiddenlayer_attn = self.forward_features(x, new_teacher_convs, is_Train)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training: # and not .jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            aug_cls = self.head(aug_x)
        return x, aug_cls, hiddenlayer_trans, hiddenlayer_attn



def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model

def student_deit_base_patch16_224_6layer(pretrained=False, **kwargs):
    model = StudentVisionTransformer(
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model
