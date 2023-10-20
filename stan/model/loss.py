import numpy as np
import math
import mindspore
from mindspore import Tensor, ops
from utils.utils import bbox_iou


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_cos(base_lr, iter, max_iter, warm_up=0.05):
    warm_up_epoch = int(max_iter * warm_up)
    if iter <= warm_up_epoch:
        lr = base_lr * (0.8 * iter / warm_up_epoch + 0.2)
    else:
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * (iter - warm_up_epoch) / (max_iter - warm_up_epoch)))
    return lr


def adjust_learning_rate(args, optimizer, i_iter):
    if args.power == -1:
        lr = lr_cos(args.lr, i_iter, args.nb_epoch)
    elif args.power != 0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    else:
        lr = args.lr * ((0.5) ** (i_iter // (args.nb_epoch // 10)))
    print(lr)
    optimizer.learning_rate = Tensor(lr)
    # optimizer.param_groups[0].learning_rate = lr
    # if len(optimizer.param_groups) > 1:
    #     optimizer.param_groups[1].learning_rate = lr / 10
    # if len(optimizer.param_groups) > 2:
    #     optimizer.param_groups[2].learning_rate = lr / 10


def yolo_loss(input, target, gi, gj, best_n_list, txt_conf, w_coord=5., w_neg=1. / 5, size_average=True, args=None):
    mseloss = mindspore.nn.MSELoss(reduction='mean')
    celoss = mindspore.nn.CrossEntropyLoss(reduction='mean')
    batch, grid = input.shape[0], input.shape[-1]

    pred_bbox = Tensor(np.zeros((batch, 4)), dtype=mindspore.float32)
    gt_bbox = Tensor(np.zeros((batch, 4)), dtype=mindspore.float32)
    for ii in range(batch):
        pred_bbox[ii, 0:2] = ops.sigmoid(input[ii, best_n_list[ii], 0:2, gj[ii], gi[ii]])
        pred_bbox[ii, 2:4] = input[ii, best_n_list[ii], 2:4, gj[ii], gi[ii]]
        gt_bbox[ii, :] = target[ii, best_n_list[ii], :4, gj[ii], gi[ii]]
    loss_x = mseloss(pred_bbox[:, 0], gt_bbox[:, 0])
    loss_y = mseloss(pred_bbox[:, 1], gt_bbox[:, 1])
    loss_w = mseloss(pred_bbox[:, 2], gt_bbox[:, 2])
    loss_h = mseloss(pred_bbox[:, 3], gt_bbox[:, 3])

    pred_conf_list, gt_conf_list = [], []
    pred_conf_list.append(input[:, :, 4, :, :].view(batch, -1))
    gt_conf_list.append(target[:, :, 4, :, :].view(batch, -1))
    pred_conf = ops.Concat(1)(pred_conf_list)
    gt_conf = ops.Concat(1)(gt_conf_list)
    gt_conf_t = gt_conf.view(batch, 9, -1).swapaxes(1, 2).view(batch, -1)
    pixel_gt = ops.true_divide(gt_conf_t.max(axis=1, return_indices=True)[1], 9).to(gt_conf.max(axis=1, return_indices=True)[1].dtype)
    loss_conf = celoss(pred_conf, gt_conf.max(axis=1, return_indices=True)[1])
    classify_loss1 = 0
    for n, i in enumerate(txt_conf[0]):
        if n < args.decomp_feat_n * args.sub_step_n and grid == 32:
            classify_loss1 += celoss(i, pixel_gt)
        elif n < 2 * args.decomp_feat_n * args.sub_step_n and n >= args.decomp_feat_n * args.sub_step_n and grid == 16:
            classify_loss1 += celoss(i, pixel_gt)
        elif n >= 2 * args.decomp_feat_n * args.sub_step_n and grid == 8:
            classify_loss1 += celoss(i, pixel_gt)
    classify_loss1 /= len(txt_conf[0])
    classify_loss2 = 0
    for n, i in enumerate(txt_conf[1]):
        classify_loss2 += ops.multilabel_soft_margin_loss(i, txt_conf[2])
    classify_loss2 /= len(txt_conf[1])
    if grid == 32:
        single_loss = celoss(txt_conf[-3], pixel_gt)
    elif grid == 16:
        single_loss = celoss(txt_conf[-2], pixel_gt)
    else:
        single_loss = celoss(txt_conf[-1], pixel_gt)

    return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf, \
           classify_loss1 * 0.5 + classify_loss2 * 0.5 + single_loss * 0.5


def build_target(raw_coord, pred, anchors_full, args):
    coord = Tensor(np.zeros((raw_coord.shape[0], raw_coord.shape[1])), dtype=mindspore.float32)
    if pred.shape[-1] == 16:
        batch, grid = raw_coord.shape[0], args.size // (args.gsize * 2)
        mask_coord = raw_coord // (args.gsize * 2)
    elif pred.shape[-1] == 8:
        batch, grid = raw_coord.shape[0], args.size // (args.gsize * 4)
        mask_coord = raw_coord // (args.gsize * 4)
    else:
        batch, grid = raw_coord.shape[0], args.size // args.gsize
        mask_coord = raw_coord // (args.gsize * 1.1)
    coord[:, 0] = (raw_coord[:, 0] + raw_coord[:, 2]) / (2 * args.size)
    coord[:, 1] = (raw_coord[:, 1] + raw_coord[:, 3]) / (2 * args.size)
    coord[:, 2] = (raw_coord[:, 2] - raw_coord[:, 0]) / args.size
    coord[:, 3] = (raw_coord[:, 3] - raw_coord[:, 1]) / args.size
    coord = coord * grid
    bbox = mindspore.Tensor(np.zeros((coord.shape[0], 9, 5, grid, grid)), dtype=mindspore.float32)

    best_n_list, best_gi, best_gj = [], [], []
    mask = mindspore.Tensor(np.zeros((coord.shape[0], grid, grid)), dtype=mindspore.float32)
    mask_coord = mask_coord.to(mindspore.int32)
    for ii in range(batch):
        gi = coord[ii, 0].long()
        gj = coord[ii, 1].long()
        tx = coord[ii, 0] - gi.float()
        ty = coord[ii, 1] - gj.float()
        gw = coord[ii, 2]
        gh = coord[ii, 3]

        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [(x[0] / (args.anchor_imsize / grid), x[1] / (args.anchor_imsize / grid)) for x in anchors]

        # Get shape of gt box
        gt_box = Tensor([[0, 0, gw.asnumpy().item(), gh.asnumpy().item()]], dtype=mindspore.float32)
        # Get shape of anchor box
        anchor_shapes = ops.Concat(1)([Tensor(np.zeros((len(scaled_anchors), 2)), dtype=mindspore.float32),
                                       Tensor(scaled_anchors, dtype=mindspore.float32)])
        # Calculate iou between gt and anchor shapes
        anch_ious = list(bbox_iou(gt_box, anchor_shapes, x1y1x2y2=False))
        # Find the best matching anchor box
        best_n = np.argmax(np.array(anch_ious))

        tw = ops.Log()(gw / scaled_anchors[best_n][0] + 1e-16)
        th = ops.Log()(gh / scaled_anchors[best_n][1] + 1e-16)

        bbox[ii, best_n, :, gj, gi] = ops.Stack()([tx, ty, tw, th, Tensor(np.ones(1), dtype=mindspore.float32).squeeze()])
        best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)
        mask[ii, mask_coord[ii][0]:mask_coord[ii][2], mask_coord[ii][1]:mask_coord[ii][3]] = 1.
    # bbox = Tensor(bbox)
    mask = mask.view(batch, -1)
    return bbox, best_gi, best_gj, best_n_list
