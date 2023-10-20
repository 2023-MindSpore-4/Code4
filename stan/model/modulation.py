from collections import OrderedDict
import math
import random
import pprint
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindspore import ops
import mindspore.nn as nn


def init_modules(modules, init='uniform'):
    if init.lower() == 'normal':
        init_params = initializer(init='HeNormal')
    elif init.lower() == 'uniform':
        init_params = initializer(init='HeUniform')
    else:
        return
    for m in modules:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Dense)):
            init_params(m.weight)


def gelu(x):
    tanh = ops.Tanh()
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * ops.Pow()(x, 3))))


class FiLM(nn.Cell):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def construct(self, x, gammas, betas):
        return (gammas * x) + betas


def mask_softmax(attn_score, word_mask, tempuature=10., clssep=False, lstm=False):
    word_mask_cp = Tensor(word_mask[:, :attn_score.shape[1]].asnumpy(), dtype=word_mask.dtype)
    score = ops.softmax(attn_score * tempuature, axis=1)
    if not clssep:
        for ii in range(word_mask_cp.shape[0]):
            if lstm:
                word_mask_cp[ii, word_mask_cp[ii, :].sum().long() - 1] = 0
            else:
                word_mask_cp[ii, 0] = 0
                word_mask_cp[ii, word_mask_cp[ii, :].sum().long()] = 0  # set one to 0 already
    mask_score = score * word_mask_cp.float()
    mask_score = mask_score / (mask_score.sum(1) + 1e-8).view(mask_score.shape[0], 1).expand(mask_score.shape[0],
                                                                                            mask_score.shape[1])
    return mask_score


class cross_att_blocked(nn.Cell):
    def __init__(self, emb_size=512, raw_feature_norm='softmax', head=3):
        super(cross_att_blocked, self).__init__()

        self.head = head
        self.raw_feature_norm = raw_feature_norm
        # visual
        self.attributes = nn.CellList()
        for n in range(head):
            self.attributes.append(nn.Dense(emb_size + 8, emb_size))

        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def construct(self, fvisu, raw_fword, coord, att_weights_txt, att_weights_visu, word_mask):
        # print("enter corss_att construct!")
        batch_size = fvisu.shape[0]
        map_fvisu_orig = fvisu
        # print("map_fvisu_orig shape: ", map_fvisu_orig.shape)
        # print("coord shape: ", coord.shape)
        map_fvisu = ops.Concat(-1)((map_fvisu_orig, coord))

        num_attributes_visu = []
        for n in range(self.head):
            map_att = self.relu(self.dropout(self.attributes[n](map_fvisu)))
            num_attributes_visu.append(map_att)

        att_attributes_visu = []
        cosine_weights_visu = []
        cosine_weights_txt = []
        att_weights_visu_collections = []

        raw_fword = l2norm(raw_fword, dim=2)

        for i, f in enumerate(num_attributes_visu):
            f = l2norm(f, dim=2)
            att_f_visu, att_weights_visu, cosine_att_visu = func_attention(f, raw_fword, self.raw_feature_norm, 9,
                                                                           weight=att_weights_visu)
            att_weights_visu_collections.append(att_weights_visu)
            att_weights_visu = ops.stack(att_weights_visu_collections, axis=-1).max(-1)
            att_attributes_visu.append(att_f_visu)
            # print("att_weights_visu shape: ", att_weights_visu.shape)
            cosine_weights_visu.append(cosine_att_visu)

        att_attributes_visu = ops.stack(att_attributes_visu, axis=-1).sum(-1)
        # print("att_attributes_visu shape: ", att_attributes_visu.shape)
        att_weights_visu_collections = ops.stack(att_weights_visu_collections, axis=-1).max(-1)
        # print("att_weights_visu_collections shape: ", att_weights_visu_collections.shape)
        num_attributes_txt = [raw_fword]
        att_attributes_txt = []
        att_weights_txt_collections = []
        map_fvisu_orig = l2norm(map_fvisu_orig, dim=2)
        for i, f in enumerate(num_attributes_txt):
            f = l2norm(f, dim=2)
            att_f_txt, att_weights_txt, cosine_att_txt = func_attention(f, map_fvisu_orig, self.raw_feature_norm, 9,
                                                                        weight=att_weights_txt)
            att_weights_txt_collections.append(att_weights_txt)
            att_weights_txt = ops.stack(att_weights_txt_collections, axis=-1).max(-1)
            # print("att_weights_txt: ", att_weights_txt.shape)
            att_attributes_txt.append(att_f_txt)
            cosine_weights_txt.append(cosine_att_txt)
        att_attributes_txt = ops.stack(att_attributes_txt,axis=-1).sum(-1)

        att_weights_txt_collections = ops.stack(att_weights_txt_collections,axis=-1).max(-1)[0]

        att_attributes_visu = l2norm(att_attributes_visu, dim=2)
        att_attributes_txt = l2norm(att_attributes_txt, dim=2)
        return att_attributes_visu, att_attributes_txt, cosine_weights_txt, cosine_weights_visu, att_weights_txt_collections, att_weights_visu_collections


class PositionEmbeddingLearned(nn.Cell):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_head=10, emb_size=512):
        super().__init__()
        self.num_pos_head = num_pos_head
        self.pos_embed = nn.Embedding(num_pos_head, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.pos_embed.embedding_table.set_data(
            initializer('uniform', [self.num_pos_head, self.pos_embed.embedding_dim]))

    def construct(self, x):
        i = ops.Range()(0, self.num_pos_head, 1)
        pos = self.pos_embed(i)
        return ops.ExpandDims()(pos, 0)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = ops.ReduceSum()(x1 * x2, dim)
    w1 = ops.l2_normalize(x1, axis=dim)
    w2 = ops.l2_normalize(x2, axis=dim)
    return (w12 / (w1 * w2).clip_by_value(min=eps))


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = ops.pow(X + eps, 2)
    norm = ops.sum(norm, dim=dim, keepdim=True)
    norm = ops.sqrt(norm) + eps
    X = ops.div(X, norm)
    return X


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = ops.abs(X)
    norm = ops.sum(norm, dim=dim, keepdim=True) + eps
    X = ops.div(X, norm)
    return X


def func_attention(query, context, raw_feature_norm, smooth=9, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.shape[0], query.shape[1]
    batch_size, sourceL = context.shape[0], context.shape[1]

    # Get attention
    # --> (batch, d, queryL)
    queryT = ops.Transpose()(query, (0, 2, 1))

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = ops.BatchMatMul()(context, queryT)

    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = ops.softmax(attn, axis=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)

    if weight is not None and weight.shape[1] == attn.shape[1] and weight.shape[2] == attn.shape[2]:
        attn = attn * (1 - weight)

    # --> (batch, queryL, sourceL)
    attn = ops.Transpose()(attn, (0, 2, 1))
    cosines_attn = ops.max(attn, axis=2)[0]
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)

    attn = ops.softmax(attn * smooth, axis=1)
    attn_out = Tensor(attn.asnumpy(), dtype=attn.dtype)
    attn_out = attn_out.view(batch_size, sourceL, queryL)

    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = ops.Transpose()(attn, (0, 2, 1))

    # --> (batch, d, sourceL)
    contextT = ops.Transpose()(context, (0, 2, 1))
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = ops.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = ops.Transpose()(weightedContext, (0, 2, 1))

    return weightedContext, attn_out, cosines_attn
