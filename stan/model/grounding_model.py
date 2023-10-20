from collections import OrderedDict

import mindspore
import numpy as np
from mindspore import Tensor
from mindspore import ops
import mindspore.nn as nn

from .darknet import *
from .convlstm import *
from .modulation import *

import argparse
import collections
import logging
import json
import re
import time
from tqdm import tqdm

from cybertron import BertModel


def st_grid_calculation(st_relevance_score, word_id_st_sent2wordlist, bbox_st_list, word_id_st_sent, st_list_bbox2word,
                        visu_scale, image_scale):
    batch_size = st_relevance_score.shape[0]
    dividend = image_scale // visu_scale
    # Tensor(np.zeros(()))
    activation_map = ops.zeros((batch_size, visu_scale, visu_scale, 1), dtype=mindspore.float32)
    for batch_i in range(batch_size):
        for ii in range(len(st_relevance_score[batch_i])):
            if not st_relevance_score[batch_i][ii] == 0:
                bbox_index = ops.nonzero(st_list_bbox2word[batch_i] == (word_id_st_sent2wordlist[batch_i][ii] + 1))
                # print("bbox_index: ", bbox_index.shape)
                for jj in bbox_index:
                    x1, y1, x2, y2 = bbox_st_list[batch_i][jj.asnumpy().item()]
                    # print("x1: ", x1)
                    grid_xl = (x1 // dividend).int().asnumpy().item()
                    grid_xr = min((x2 // dividend + 1).int().asnumpy().item(), visu_scale - 1)
                    grid_yt = (y1 // dividend).int().asnumpy().item()
                    grid_yb = min((y2 // dividend + 1).int().asnumpy().item(), visu_scale - 1)
                    activation_map[batch_i, grid_yt:grid_yb, grid_xl:grid_xr] = st_relevance_score[batch_i][
                        ii].asnumpy().item()
    # grid softmax
    for batch_i in range(batch_size):
        if not len(ops.nonzero(activation_map[batch_i])) == 0:
            tmp = activation_map[batch_i]
            tmp = tmp.reshape(-1, 1)
            tmp = nn.Softmax()(tmp * 9)
            tmp = tmp.reshape(visu_scale, visu_scale, -1)
            activation_map[batch_i] = tmp
    return activation_map


def generate_coord(batch, height, width):
    xv, yv = ops.meshgrid(ops.arange(0, height), ops.arange(0, width))
    xv_min = (xv.astype('float32') * 2 - width) / width
    yv_min = (yv.astype('float32') * 2 - height) / height
    xv_max = ((xv + 1).astype('float32') * 2 - width) / width
    yv_max = ((yv + 1).astype('float32') * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = ops.ones((height, width), dtype=mindspore.float32) * (1. / height)
    wmap = ops.ones((height, width), dtype=mindspore.float32) * (1. / width)
    coord = Tensor(ops.Concat(0)((xv_min.unsqueeze(0), yv_min.unsqueeze(0),
                                  xv_max.unsqueeze(0), yv_max.unsqueeze(0),
                                  xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),
                                  hmap.unsqueeze(0), wmap.unsqueeze(0))))
    coord = ops.ExpandDims()(coord, 0)
    coord = ops.Tile()(coord, (batch, 1, 1, 1))
    return coord


class cross_attention_head(nn.Cell):
    def __init__(self, emb_size=256, tunebert=False, convlstm=False, bert_model='bert-base-uncased', leaky=False,
                 jemb_drop_out=0.1, raw_feature_norm='softmax', NCatt=2, down_sample_ith=2, fpn_n=3, sub_step=2,
                 n_head=3):
        super(cross_attention_head, self).__init__()

        self.down_sample_ith = down_sample_ith
        self.fpn_n = fpn_n
        self.emb_size = emb_size
        self.NCatt = NCatt
        self.sub_step = sub_step
        self.tunebert = tunebert
        self.raw_feature_norm = raw_feature_norm
        if bert_model == 'bert-base-uncased':
            self.textdim = 768
        else:
            self.textdim = 1024
        self.convlstm = convlstm
        ## Visual model
        self.visumodel = Darknet(config_path='./model/yolov3.cfg')
        self.visumodel.load_weights('./saved_models/yolov3.weights')
        # Text model
        self.textmodel = BertModel.load(bert_model)
        self.mapping_visu = ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky)
        self.mapping_lang = nn.SequentialCell(
            nn.Dense(self.textdim, emb_size),
            nn.ReLU(),
            nn.Dropout(p=jemb_drop_out),
            nn.Dense(emb_size, emb_size),
            nn.ReLU(), )
        self.txt_single_classifier = ConvBatchNormReLU(emb_size * 2, 1, 1, 1, 0, 1, leaky=leaky)
        self.softmax = nn.Softmax()
        self.bn = nn.BatchNorm2d(emb_size)
        self.cross_att_cell = nn.CellList()
        self.name2index = OrderedDict()
        output_emb = emb_size
        modules = OrderedDict()
        self.name2index['convmerge0_1x1'] = 0
        self.cross_att_cell.append(ConvBatchNormReLU(emb_size * 2, emb_size, 1, 1, 0, 1))
        self.name2index['convmerge1_1x1'] = 1
        self.cross_att_cell.append(ConvBatchNormReLU(emb_size * 2, emb_size, 1, 1, 0, 1))
        count = len(self.name2index)
        for i in range(fpn_n - 1):
            self.name2index['conv%d_downsample' % i] = count + i
            conv_cell = nn.SequentialCell(
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1),
                nn.MaxPool2d(2, 2))
            self.cross_att_cell.append(conv_cell)
        count = len(self.name2index)
        self.name2index.update({'fcn': count, 'fcn_sub': count + 1, 'fcn_2sub': count + 2})
        fcn_cell = nn.SequentialCell(
            ConvBatchNormReLU(output_emb * 2, output_emb, 1, 1, 0, 1, leaky=leaky),
            nn.Conv2d(output_emb, 9 * 5, kernel_size=1, has_bias=True, pad_mode='pad'))
        self.cross_att_cell.append(fcn_cell)
        fcn_sub_cell = nn.SequentialCell(
            ConvBatchNormReLU(output_emb * 2, output_emb, 1, 1, 0, 1, leaky=leaky),
            nn.Conv2d(output_emb, 9 * 5, kernel_size=1, has_bias=True, pad_mode='pad'))
        self.cross_att_cell.append(fcn_sub_cell)
        fcn_2sub_cell = nn.SequentialCell(
            ConvBatchNormReLU(output_emb * 2, output_emb, 1, 1, 0, 1, leaky=leaky),
            nn.Conv2d(output_emb, 9 * 5, kernel_size=1, has_bias=True, pad_mode='pad'))
        self.cross_att_cell.append(fcn_2sub_cell)
        count = len(self.name2index)
        kn = 0
        for _ in range(0, self.fpn_n):
            for _ in range(self.sub_step):
                self.name2index.update({'catt%d' % kn: count, 'linear%d' % kn: count + 1, 'conv%d_1x1' % kn: count + 2,
                                        'conv%d_3x3' % kn: count + 3})
                # print("before cross_att construct!")
                self.cross_att_cell.append(cross_att_blocked(raw_feature_norm=raw_feature_norm, head=n_head))
                self.cross_att_cell.append(nn.Dense(1024, emb_size))
                self.cross_att_cell.append(ConvBatchNormReLU(emb_size * 2, emb_size, 1, 1, 0, 1))
                self.cross_att_cell.append(ConvBatchNormReLU(emb_size, emb_size, 3, 1, 1, 1))
                count += 4
                kn += 1

    def construct(self, image, word_id, word_mask, word_st_position, bbox_st_list, word_id_st_sent, word_mask_st_sent,
                  st_list_bbox2word):

        # Visual Module
        # print("image type: ", type(image))
        batch_size = image.shape[0]
        raw_fvisu = self.visumodel(image)
        # print("test here!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.convlstm:
            # print("not enter raw_fvisu_8*8")
            raw_fvisu = raw_fvisu[1]
        else:
            # print("enter raw_fvisu_8*8")
            raw_fvisu_8x8 = raw_fvisu[0]
            # print("11111raw_fvisu_8x8: ", raw_fvisu_8x8.shape)
            raw_fvisu_16x16 = raw_fvisu[1]
            raw_fvisu = raw_fvisu[2]

        # Language Module for scene text
        all_encoder_layers_st_sent, _ = self.textmodel(word_id_st_sent, token_type_ids=None,
                                                       attention_mask=word_mask_st_sent)
        # print("word_id_st_sent: ", word_id_st_sent.shape)
        # print("attention_mask: ", word_mask_st_sent.shape)
        # print("all_encoder_layers_st_sent: ", all_encoder_layers_st_sent.shape)
        raw_flang_st_sent = all_encoder_layers_st_sent
        raw_fword_st_sent = all_encoder_layers_st_sent
        if not self.tunebert:
            hidden_st_sent = raw_flang_st_sent
            raw_fword_st_sent.stop_gradient = True

        # Language Module for expression
        all_encoder_layers, _ = self.textmodel(word_id,
                                               token_type_ids=None, attention_mask=word_mask)
        # Sentence feature at the first position [cls]
        raw_flang = all_encoder_layers
        raw_fword = all_encoder_layers
        if not self.tunebert:
            # fix bert during training
            # raw_flang = raw_flang.detach()
            hidden = raw_flang
            raw_fword.stop_gradient = True

        # Correlatd Text Extraction & Correlated Region Activation
        mask_word_att = ops.ZerosLike()(raw_fword)
        mask_st_att = ops.ZerosLike()(raw_fword_st_sent)
        for ii in range(batch_size):
            mask_word_att[ii, 1:len(ops.NotEqual()(word_mask[ii], 0)) - 1, :] = 1
            mask_st_att[ii, 1:len(ops.NotEqual()(word_id_st_sent[ii], 0)) - 1, :] = 1
        raw_fword_attn = raw_fword * mask_word_att
        raw_fword_st_sent = raw_fword_st_sent * mask_st_att
        st_relevance_score = Tensor(np.zeros((batch_size, mask_st_att.shape[1], mask_word_att.shape[1])),
                                    dtype=mindspore.float32)

        THRES_PHI = 0.50
        for ii in range(batch_size):
            st_relevance_score[ii] = ops.cosine_similarity(raw_fword_st_sent[ii].unsqueeze(1), raw_fword_attn[ii],
                                                           dim=-1)
        # print("st_relevance_score: ", st_relevance_score.shape)
        # print("image shape: ", image.shape)
        st_relevance_score = ops.ReduceMax(keep_dims=True)(st_relevance_score, 2)
        st_relevance_score = ops.Select()(st_relevance_score < THRES_PHI, ops.ZerosLike()(st_relevance_score),
                                          st_relevance_score)
        # print("st_relevance_score: ", st_relevance_score.shape)
        weighted_st_feature_8x8 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list,
                                                      word_id_st_sent, st_list_bbox2word, raw_fvisu_8x8.shape[2],
                                                      image.shape[2])
        raw_fvisu_8x8 = raw_fvisu_8x8.transpose(0, 2, 3, 1) * weighted_st_feature_8x8 + raw_fvisu_8x8.transpose(0, 2, 3,
                                                                                                                1)
        raw_fvisu_8x8 = raw_fvisu_8x8.transpose(0, 3, 1, 2)

        weighted_st_feature_16x16 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list,
                                                        word_id_st_sent, st_list_bbox2word, raw_fvisu_16x16.shape[2],
                                                        image.shape[2])
        raw_fvisu_16x16 = raw_fvisu_16x16.transpose(0, 2, 3, 1) * weighted_st_feature_16x16 + raw_fvisu_16x16.transpose(
            0, 2, 3, 1)
        raw_fvisu_16x16 = raw_fvisu_16x16.transpose(0, 3, 1, 2)

        weighted_st_feature_32x32 = st_grid_calculation(st_relevance_score, word_st_position, bbox_st_list,
                                                        word_id_st_sent, st_list_bbox2word, raw_fvisu.shape[2],
                                                        image.shape[2])
        raw_fvisu = raw_fvisu.transpose(0, 2, 3, 1) * weighted_st_feature_32x32 + raw_fvisu.transpose(0, 2, 3, 1)
        raw_fvisu = raw_fvisu.transpose(0, 3, 1, 2)

        # Language Module - mapping language feature
        fword = Tensor(np.zeros((raw_fword.shape[0], raw_fword.shape[1], self.emb_size)), dtype=mindspore.float32)
        l2_normalize = ops.L2Normalize(axis=1)
        for ii in range(raw_fword.shape[0]):
            ntoken = (word_mask[ii] != 0).sum()
            fword[ii, :ntoken, :] = l2_normalize(self.mapping_lang(raw_fword[ii][:ntoken]))
        raw_fword = fword
        global_raw_fword = ops.ReduceMean()(raw_fword, axis=1)

        ## Visual Module - mapping visual feature & decomposition
        fvisu = self.mapping_visu(raw_fvisu)
        raw_fvisu = l2_normalize(fvisu)  # 32x32
        raw_fvisu_16x16 = l2_normalize(raw_fvisu_16x16)  # 16x16
        # print("1raw_fvisu_8x8: ", raw_fvisu_8x8.shape)
        raw_fvisu_8x8 = raw_fvisu_8x8.view(batch_size, raw_fvisu_8x8.shape[1], -1).swapaxes(1, 2)
        maxpool_1d = nn.MaxPool1d(2, 2)
        raw_fvisu_8x8 = maxpool_1d(raw_fvisu_8x8).swapaxes(1, 2).view(batch_size, -1, 8, 8)
        raw_fvisu_8x8 = l2_normalize(raw_fvisu_8x8)  # 8x8
        # print("raw_fvisu_8x8: ", raw_fvisu_8x8.shape)

        map_fvisu = raw_fvisu.view(batch_size, raw_fvisu.shape[1], -1)
        map_fvisu_orig = map_fvisu.swapaxes(1, 2)
        map_fvisu_16x16 = raw_fvisu_16x16.view(batch_size, raw_fvisu_16x16.shape[1], -1)
        map_fvisu_16x16 = map_fvisu_16x16.swapaxes(1, 2)
        map_fvisu_8x8 = raw_fvisu_8x8.view(batch_size, raw_fvisu_8x8.shape[1], -1)
        # print("map_fvisu_8x8: ", map_fvisu_8x8.shape)
        map_fvisu_8x8 = map_fvisu_8x8.swapaxes(1, 2)
        # print("map_fvisu_8x8: ", map_fvisu_8x8.shape)
        map_fvisu_orig_co = Tensor(map_fvisu_orig.asnumpy(), dtype=map_fvisu_orig.dtype)

        ## Visual Module - location feature
        coord = generate_coord(batch_size, raw_fvisu.shape[2], raw_fvisu.shape[3])
        coord_16x16 = generate_coord(batch_size, raw_fvisu_16x16.shape[2], raw_fvisu_16x16.shape[3])
        coord_8x8 = generate_coord(batch_size, raw_fvisu_8x8.shape[2], raw_fvisu_8x8.shape[3])

        map_coord = coord.view(batch_size, coord.shape[1], -1)
        map_coord = map_coord.swapaxes(1, 2)
        map_coord_16x16 = coord_16x16.view(batch_size, coord_16x16.shape[1], -1)
        map_coord_16x16 = map_coord_16x16.swapaxes(1, 2)
        map_coord_8x8 = coord_8x8.view(batch_size, coord_8x8.shape[1], -1)
        map_coord_8x8 = map_coord_8x8.swapaxes(1, 2)

        ## Initialization for bottom-up and bidirectional fusion
        make_f = []
        make_target_visu = []
        make_target_txt = []
        out_feat = []
        cosine_weights = []
        contrast_visu = 0
        contrast_txt = 0
        cosine_txt_word, cosine_txt_visu = None, None
        map_fvisu_add = map_fvisu_orig
        map_coord_add = map_coord
        raw_fvisu_add = raw_fvisu
        out_visu = 0
        merge_t = 0
        att_n = 0

        for ff in range(self.fpn_n):  # for multi-scale visual features
            for n in range(self.sub_step):  # for multi-step alignment
                if ff != 0 or n != 0:
                    # print("enter merge_f test!")
                    out_visu = merge_f.view(batch_size, raw_fvisu.shape[1], -1)
                    out_visu = out_visu.swapaxes(1, 2)
                # print("ff, n: ", ff, n)
                # print("word_mask.shape: ", word_mask.shape)
                # print("map_fvisu_add shape: ", map_fvisu_add.shape)
                # print("net: ", self.cross_att_cell[self.name2index['catt%d' % att_n]])
                out_visu, out_txt, cosine_txt_region, cosine_visu_region, cosine_txt_word, cosine_txt_visu = \
                    self.cross_att_cell[self.name2index['catt%d' % att_n]](out_visu + map_fvisu_add,
                                                                           merge_t + raw_fword,
                                                                           map_coord_add, cosine_txt_word,
                                                                           cosine_txt_visu,
                                                                           word_mask)

                out_visu = out_visu + global_raw_fword.unsqueeze(1)
                # print("out_visu: ", out_visu.shape)

                out_visu = out_visu.swapaxes(1, 2)
                out_visu = out_visu.view(batch_size, raw_fvisu_add.shape[1], raw_fvisu_add.shape[2],
                                         raw_fvisu_add.shape[3])
                merge_f = ops.Concat(axis=1)((raw_fvisu_add + contrast_visu, out_visu))
                merge_f = self.cross_att_cell[self.name2index['conv%d_1x1' % att_n]](merge_f)
                merge_f = self.cross_att_cell[self.name2index['conv%d_3x3' % att_n]](merge_f)
                merge_t = ops.Concat(axis=-1)((raw_fword + contrast_txt, out_txt))
                merge_t = self.cross_att_cell[self.name2index['linear%d' % att_n]](merge_t)
                make_target_visu.extend(cosine_visu_region)
                make_target_txt.extend(cosine_txt_region)
                make_f.append(merge_f)
                att_n += 1
            if ff == 0:
                max_feature_32x32 = ops.stack(make_f[:self.sub_step], -1).sum(-1)
                merge_f = self.cross_att_cell[self.name2index['conv0_downsample']](max_feature_32x32)
                raw_fvisu_add = raw_fvisu_16x16
                map_fvisu_add = map_fvisu_16x16
                # print("ff==0. map_fvisu_add shape: ", map_fvisu_add.shape)
                map_coord_add = map_coord_16x16

            elif ff == 1:
                max_feature_16x16 = ops.stack(make_f[self.sub_step:self.sub_step * (ff + 1)], -1).sum(-1)
                merge_f = self.cross_att_cell[self.name2index['conv1_downsample']](max_feature_16x16)
                raw_fvisu_add = raw_fvisu_8x8
                map_fvisu_add = map_fvisu_8x8
                # print("ff==1. map_fvisu_add shape: ", map_fvisu_add.shape)
                map_coord_add = map_coord_8x8
            if ff == self.fpn_n - 1 and n == self.sub_step - 1:
                # print("enter fpn_n", ff, n)
                max_feature_8x8 = ops.stack(make_f[self.sub_step * ff:], -1).sum(-1)
                # print("max_feature_8x8: ", max_feature_8x8.shape)
                fpn_region_16x16 = ops.interpolate(max_feature_8x8, scale_factor=2., recompute_scale_factor=True, mode="bilinear", align_corners=True)
                fpn_region_16x16 = ops.Concat(axis=1)((max_feature_16x16, fpn_region_16x16))
                fpn_region_16x16 = self.cross_att_cell[self.name2index['convmerge0_1x1']](fpn_region_16x16)
                fpn_region_32x32 = ops.interpolate(fpn_region_16x16, scale_factor=2., recompute_scale_factor=True, mode="bilinear", align_corners=True)
                fpn_region_32x32 = ops.Concat(axis=1)((max_feature_32x32, fpn_region_32x32))
                fpn_region_32x32 = self.cross_att_cell[self.name2index['convmerge1_1x1']](fpn_region_32x32)
                out_region_32x32 = self.cross_att_cell[self.name2index['fcn']](
                    ops.Concat(axis=1)((fpn_region_32x32, raw_fvisu)))
                out_region_16x16 = self.cross_att_cell[self.name2index['fcn_sub']](
                    ops.Concat(axis=1)((fpn_region_16x16, raw_fvisu_16x16)))
                out_region_8x8 = self.cross_att_cell[self.name2index['fcn_2sub']](
                    ops.Concat(axis=1)((max_feature_8x8, raw_fvisu_8x8)))
        single_conf = self.txt_single_classifier(ops.Concat(axis=1)((max_feature_32x32, raw_fvisu))).view(batch_size,
                                                                                                          map_fvisu_orig_co.shape[
                                                                                                              1])
        # print("single_conf: ", single_conf.shape)
        single_conf_16 = self.txt_single_classifier(ops.Concat(axis=1)((max_feature_16x16, raw_fvisu_16x16))).view(
            batch_size, 16*16)
        # print("single_conf_16: ", single_conf_16.shape)
        # print("max_feature_8x8: ", max_feature_8x8.shape, type(max_feature_8x8), max_feature_8x8.dtype)
        # print("raw_fvisu_8x8: ", raw_fvisu_8x8.shape, type(raw_fvisu_8x8), raw_fvisu_8x8.dtype)
        # print(self.txt_single_classifier)
        concat_max_raw = ops.cat((max_feature_8x8, raw_fvisu_8x8), axis=1)
        # print("concat_max_raw: ", concat_max_raw.shape)
        single_conf_8 = self.txt_single_classifier(concat_max_raw)
        single_conf_8 = single_conf_8.view(batch_size, 8 * 8)
        # print("after single_conf_8")
        out_feat.extend([out_region_32x32, out_region_16x16, out_region_8x8])
        cosine_weights.extend(
            [make_target_visu, make_target_txt, word_mask, single_conf, single_conf_16, single_conf_8])

        return out_feat, cosine_weights


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = ops.ReduceSum()(x1 * x2, dim)
    w1 = ops.L2Normalize(axis=dim)(x1)
    w2 = ops.L2Normalize(axis=dim)(x2)
    return (w12 / (w1 * w2).clip_by_value(min=eps))
