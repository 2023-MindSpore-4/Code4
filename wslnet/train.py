#!/usr/bin/python3
#coding=utf-8

import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import datetime
import cv2
from Network.rnet_up import WSLnet_up
from Network.rnet_down import WSLNet_down
from Network.snet import Snet
import mindspore
import logging as logger
from mindspore.dataset import GeneratorDataset
from data import dataset
from mindspore import nn
import numpy as np
from mindspore import dtype as mstype
from mindspore import Tensor
import lib as sche

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True
IMAGE_GROUP = 0
mindspore.set_context(device_target='GPU', device_id=0)

class CrossEntropyLossCell_rnet(nn.Cell):
    def __init__(self, backbone1, backbone2, loss_fn):
        super(CrossEntropyLossCell_rnet, self).__init__(auto_prefix=False)
        self._backbone1 = backbone1
        self._backbone2 = backbone2
        self._loss_fn = loss_fn

    def construct(self, data, coarse, label):
        out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = self._backbone2(data)
        # out2, out3, out4, out5 = self._backbone1(data, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)
        out2, out3, out4, out5 = self._backbone1(data, coarse, down_out1)
        loss2 = self._loss_fn(out2, label)
        loss3 = self._loss_fn(out3, label)
        loss4 = self._loss_fn(out4, label)
        loss5 = self._loss_fn(out5, label)
        # loss2c = self._loss_fn(out2_c, label)
        # loss3c = self._loss_fn(out3_c, label)
        # loss4c = self._loss_fn(out4_c, label)
        # loss5c = self._loss_fn(out5_c, label)
        # loss6 = loss2c * 1 + loss3c * 0.8 + loss4c * 0.6 + loss5c * 0.4
        loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
        return loss


class RNetTrainOneStepCell(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer):
        """入参有两个：训练网络，优化器"""
        super(RNetTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                           # 定义前向网络
        self.network.set_grad()                          # 构建反向网络
        self.optimizer = optimizer                       # 定义优化器
        self.weights = self.optimizer.parameters         # 待更新参数
        self.grad = mindspore.ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        loss = self.network(*inputs)                            # 计算当前输入的损失函数值
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        self.optimizer(grads)                                   # 使用优化器更新权重参数
        return loss

def train(Dataset, Network1, Network2, Network3):
    column_name = ["image", "mask", "coarse", "H", "W", "name"]
    # column_name = ["image", "mask", "coarse"]
    ## dataset GCPANet
    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=8, lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = GeneratorDataset(data, shuffle=True, column_names=column_name, num_parallel_workers=8).batch(cfg.batch)

    ## dataset FCRNet
    cfg2 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=8,
                          lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
    data2 = Dataset.Data(cfg2)
    loader2 = GeneratorDataset(data2, column_names=column_name, shuffle=True, num_parallel_workers=8).batch(cfg2.batch)

    ## network
    net1 = Network1()
    net2 = Network2()
    net = Network3()

    ## parameter
    base, head = [], []
    scheduler = nn.piecewise_constant_lr(milestone=[10, 20], learning_rates=[cfg.lr * 0.1, cfg.lr * 0.01])


    optimizer2 = nn.Adam(net.trainable_params(), learning_rate=scheduler, weight_decay=cfg.decay)

    optimizer3 = nn.SGD(net1.trainable_params(), learning_rate=scheduler, momentum=cfg2.momen, weight_decay=cfg2.decay, nesterov=True)

    optimizer1 = nn.Adam(net2.trainable_params(), learning_rate=scheduler, weight_decay=cfg2.decay)

    global_step = 0
    mae_global = 1
    mae_snet = 1

    for epoch_g in range(5):
        if epoch_g == 0:
            mode_refine1 = 'train_0'
            mode_refine2 = 'train_0'
        elif epoch_g == 1:
            mode_refine1 = 'train_2'
            mode_refine2 = 'train_0'
        elif epoch_g == 2:
            mode_refine1 = 'train_2_4'
            mode_refine2 = 'train_0'
        elif epoch_g == 3:
            mode_refine1 = 'train_2_4_6'
            mode_refine2 = 'train_0'
        elif epoch_g == 4:
            mode_refine1 = 'train_2_4_6_8'
            mode_refine2 = 'train_0'
        elif epoch_g == 5:
            mode_refine1 = 'train_2_4_6_8_10'
            mode_refine2 = 'train_0'
        else:
            mode_refine1 = 'train_2_4_6_8_9'
            mode_refine2 = 'train_0'
        cfg1 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_refine1, batch=2,
                             lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
        data1 = Dataset.Data(cfg1)
        loader1 = GeneratorDataset(data1, shuffle=True, column_names=column_name, num_parallel_workers=1).batch(cfg1.batch)

        cfg2 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_refine2, batch=2,
                              lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
        data2 = Dataset.Data(cfg2)
        loader2 = GeneratorDataset(data2, shuffle=True, column_names=column_name, num_parallel_workers=1).batch(cfg2.batch)

        data_iterator1 = loader1.create_dict_iterator()

        data_iterator2 = loader2.create_dict_iterator()

        loss = nn.BCEWithLogitsLoss()
        net_with_loss = CrossEntropyLossCell_rnet(net1, net2, loss_fn=loss)
        train_net = RNetTrainOneStepCell(net_with_loss, optimizer1)
        train_net.set_train()

        for epoch in range(cfg1.epoch):
            batch_idx = -1
            logger.info(cfg1.mode)
            for iteration, data in enumerate(data_iterator1, start=1):
                batch_idx += 1
                global_step += 1
                image = Tensor(data["image"], mstype.float32)
                coarse = data["coarse"]
                mask = data["mask"]
                # image = data["image"]
                # coarse = Tensor(data["coarse"], mstype.float64)
                # mask = Tensor(data["mask"], mstype.float64)
                logger.info(image.shape)
                logger.info(coarse.shape)
                logger.info(mask.shape)
                loss = train_net(image, coarse, mask)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d loss=%.6f ' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g+1,
                    loss.item(0))
                    print(msg)
                    logger.info(msg)
            for iteration, data in data_iterator2:
                batch_idx += 1
                global_step += 1
                loss = train_net(data["image"], data["coarse"], data["mask"])
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f ' % (
                        datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g + 1,
                        optimizer1.param_groups[0]['lr'],
                        loss.item())
                    print(msg)
                    logger.info(msg)
            if (epoch + 1) % 10 == 0:
                mindspore.save_checkpoint(net1,
                           cfg.savepath + '/model-refine-up-' + str(epoch_g) + '-' + str(epoch + 1) + '.ckpt')
                mindspore.save_checkpoint(net2,
                           cfg.savepath + '/model-refine-down-' + str(epoch_g) + '-' + str(epoch + 1) + '.ckpt')

        # test daaset
        cfg_mae = Dataset.Config(datapath='./data/DUTS', mode='test0')
        data_mae = Dataset.Data(cfg_mae)
        loader_mae = GeneratorDataset(data_mae, shuffle=True, column_names=column_name, num_parallel_workers=8).batch(1)
        net_mae1 = net1
        net_mae1.set_train(mode=False)
        net_mae2 = net2
        net_mae2.set_train(mode=False)
        data_iterator = loader_mae.create_tuple_iterator()

        with mindspore.ops.stop_gradient():
            mae, cnt = 0, 0
            for image, mask, coarse, (H, W), name in data_iterator:
                image, coarse, mask = image.float(), coarse.float(), mask.float()
                out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = net2(image)
                out2, out3, out4, out5 = net1(image, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)
                pred = mindspore.ops.Sigmoid(out2)

                # mae
                cnt += 1
                mae += (pred - mask).abs().mean()
            mae = mae / cnt
            mae = mae.item()
            print("mae_snet=", mae)
            logger.info("mae_snet=" + str(mae))

            if mae > mae_global:
                if epoch_g == 0:
                    refine_test = 'test_0'
                elif epoch_g == 1:
                    refine_test = 'test_3'
                elif epoch_g == 2:
                    refine_test = 'test_5'
                elif epoch_g == 3:
                    refine_test = 'test_7'
                elif epoch_g == 4:
                    refine_test = 'test_9'
                else:
                    refine_test = 'test_10'

                param_net1 = mindspore.load_checkpoint("ours/model-refine-up-" + str(epoch_g - 1) + "-30.ckpt")
                mindspore.load_param_into_net(net1, param_net1)
                param_net2 = mindspore.load_checkpoint("ours/model-refine-down-" + str(epoch_g-1) + "-30.ckpt")
                mindspore.load_param_into_net(net2, param_net2)
                print("model not change")
                logger.info("model not change")
            else:
                if epoch_g == 0:
                    refine_test = 'test_0'
                elif epoch_g == 1:
                    refine_test = 'test_1_3'
                elif epoch_g == 2:
                    refine_test = 'test_1_3_5'
                elif epoch_g == 3:
                    refine_test = 'test_1_3_5_7'
                elif epoch_g == 4:
                    refine_test = 'test_1_3_5_7_9'
                else:
                    refine_test = 'test_1_3_5_7_9_10'
                mae_global = mae
                print("model change")
                logger.info("model change")

        cfg_test = Dataset.Config(datapath='./data/DUTS', mode=refine_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = GeneratorDataset(data_test, column_names=column_name, shuffle=True, num_parallel_workers=8).batch(1)
        # test network
        net_test1 = net1
        net_test1.set_train(mode=False)

        net_test2 = net2
        net_test2.set_train(mode=False)

        with mindspore.ops.stop_gradient():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image, coarse = image.float(), coarse.float()

                out2_c, out3_c, out4_c, out5_c, down_out1, down_out2, down_out3, down_out4, down_out5 = net2(image)
                out2, out3, out4, out5 = net1(image, coarse, down_out1, down_out2, down_out3, down_out4, down_out5)
                out2 = mindspore.ops.interpolate(out2, sizes=(H, W), mode="bilinear")
                pred = (mindspore.ops.Sigmoid(out2[0, 0]) * 255).asnumpy()
                head = './data/DUTS/noise' + str(epoch_g)
                if not os.path.exists(head):
                    os.makedirs(head)

                cv2.imwrite(head + '/' + name[0], np.uint8(pred))
                name = name[0].split('.')
                if i == 0:
                    cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_refine.png',
                                np.uint8(pred))
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '.png', np.uint8(pred))
                i += 1


        ## dataset MINet
        if epoch_g == 0:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1'
        elif epoch_g == 1:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3'
        elif epoch_g == 2:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_5'
        elif epoch_g == 3:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_5_7'
        elif epoch_g == 4:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9'
        elif epoch_g == 5:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9_10'
        else:
            mode_sod0 = 'train1_0'
            mode_sod1 = 'train1_1_3_3_5_7_9_10'

        cfg0 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_sod0, batch=12, lr=1e-4, momen=0.9,
                              decay=5e-4, epoch=30)
        data0 = Dataset.Data(cfg0)
        loader0 = GeneratorDataset(data0, column_names=column_name, shuffle=True, num_parallel_workers=4).batch(cfg0.batch)

        cfg1 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_sod1, batch=12, lr=1e-4, momen=0.9,
                              decay=5e-4, epoch=30)
        data1 = Dataset.Data(cfg1)
        loader1 = GeneratorDataset(data1, column_names=column_name, shuffle=True, num_parallel_workers=4).batch(cfg1.batch)

        loss_fn = mindspore.nn.BCEWithLogitsLoss
        net_with_loss = nn.WithLossCell(net, loss_fn=loss_fn)
        train_network = nn.TrainOneStepCell(net_with_loss, optimizer2)
        train_network.set_train()
        data_iterator0 = loader0.create_tuple_iterator()
        data_iterator1 = loader1.create_tuple_iterator()
        for epoch in range(cfg0.epoch):
            for image, mask, coarse, (H, W), name in data_iterator1:
                global_step += 1
                batch_idx += 1
                loss = train_network(image, mask)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g + 1,
                    optimizer2.param_groups[0]['lr'], loss.item())
                    print(msg)
                    logger.info(msg)
            for image, mask, coarse, (H, W), name in data_iterator0:
                global_step += 1
                batch_idx += 1
                loss = train_network(image, mask)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g + 1,
                    optimizer2.param_groups[0]['lr'], loss.item())
                    print(msg)
                    logger.info(msg)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epoch:
                mindspore.save_checkpoint(net1,
                                          cfg.savepath + '/model-sod-' + str(epoch_g) + '-' + str(epoch + 1) + '.ckpt')

        # test dataset
        cfg_mae = Dataset.Config(datapath='./data/DUTS', mode='test0')
        data_mae = Dataset.Data(cfg_mae)
        loader_mae = GeneratorDataset(data_mae, column_names=column_name, shuffle=True, num_parallel_workers=8).batch(1)
        net_mae = net
        net_mae.set_train(mode=False)

        with mindspore.ops.stop_gradient():
            mae, cnt = 0, 0
            for image, mask, coarse, (H, W), name in loader_mae:
                image, coarse, mask = image.float(), coarse.float(), mask.float()
                pred_mask = net_mae(image)
                pred = mindspore.ops.Sigmoid(pred_mask)

                # mae
                cnt += 1
                mae += (pred - mask).abs().mean()
            mae = mae / cnt
            mae = mae.item()
            print("mae_snet=", mae)
            logger.info("mae_snet=" + str(mae))

        if mae > mae_snet:
                if epoch_g == 0:
                    sod_test = 'test_2'
                elif epoch_g == 1:
                    sod_test = 'test_4'
                elif epoch_g == 2:
                    sod_test = 'test_6'
                elif epoch_g == 3:
                    sod_test = 'test_8'
                elif epoch_g == 4:
                    sod_test = 'test_10'
                else:
                    sod_test = 'test_10'

                param_net = mindspore.load_checkpoint("ours/model-sod-" + str(epoch_g - 1) + "-30.ckpt")
                mindspore.load_param_into_net(net, param_net)

                print("model not change")
                logger.info("model not change")
        else:
            if epoch_g == 0: sod_test = 'test_2'
            elif epoch_g == 1: sod_test = 'test_2_4'
            elif epoch_g == 2: sod_test = 'test_2_4_6'
            elif epoch_g == 3:
                sod_test = 'test_2_4_6_8'
            elif epoch_g == 4:
                sod_test = 'test_2_4_6_8_10'
            mae_snet = mae
            print("model change")
            logger.info("model change")

        cfg_test = Dataset.Config(datapath='./data/DUTS', mode=sod_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = GeneratorDataset(data_test, column_names=column_name, shuffle=True, num_parallel_workers=8).batch(1)
        # test network
        net_test = net

        with mindspore.ops.stop_gradient():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image = image.cuda().float()
                pred_mask = net_test(image)
                pred_mask = mindspore.ops.interpolate(pred_mask, sizes=(H, W), mode='bilinear')
                pred = (mindspore.ops.Sigmoid(pred_mask[0, 0]) * 255).asnumpy()
                # print(name[0])
                name = name[0].split('.')
                head = './data/DUTS/coarse_sod_' + str(epoch_g)
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.uint8(pred))
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '.png', np.uint8(pred))
                if i == 0:
                    cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_sod.png', np.uint8(pred))
                i += 1


if __name__=='__main__':
    train(dataset, WSLnet_up, WSLNet_down, Snet)