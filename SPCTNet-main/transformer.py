from mindspore import nn, ops
from einops import rearrange


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Dense(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def construct(self, x):
        return self.net(x)


class PatchSelfAttention(nn.Cell):
    """
    dim:
        Token's dimension, EX: word embedding vector size
    """
    def __init__(self, dim):
        super(PatchSelfAttention, self).__init__()
        self.to_qvk = nn.Dense(dim, dim * 3, has_bias=False)
        self.scale_factor = dim ** -0.5

        # Final linear transformation layer
        self.w_out = nn.Dense(dim, dim, has_bias=False)

    def construct(self, x):
        q, k, v = self.to_qvk(x).chunk(3, dim=-1)

        dots = ops.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = ops.softmax(dots, axis=-1)
        out = ops.matmul(attn, v)
        return self.w_out(out)


class Transformer(nn.Cell):
    def __init__(self, dim, depth, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                PreNorm(dim, PatchSelfAttention(dim)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerBlock(nn.Cell):
    def __init__(self, in_channels, layers, mlp_dim, patch_size=4, drop_out=0.0):
        super(TransformerBlock, self).__init__()
        self.patch_size = patch_size
        self.transformer = Transformer(in_channels, layers, mlp_dim=mlp_dim, dropout=drop_out)

    def construct(self, x):
        _, _, h, w, s = x.shape
        global_repr = rearrange(x, 'b d (h ph) (w pw) (s ps)  -> b (ph pw ps) (h w s) d',
                                ph=self.patch_size, pw=self.patch_size,ps=self.patch_size)
        global_repr = self.transformer(global_repr)
        global_repr = rearrange(global_repr, 'b (ph pw ps) (h w s) d -> b d (h ph) (w pw) (s ps)',
                                h=h // self.patch_size, w=w // self.patch_size,s= s//self.patch_size,
                                ph=self.patch_size, pw=self.patch_size,ps=self.patch_size)
        return global_repr


if __name__ == '__main__':
    x = ops.randn(2,64,24,48,48)
    net = TransformerBlock(in_channels=64,layers=2,mlp_dim=128)
    y = net(x)
    print(y.shape)