import mindspore
import mindspore as ms
from mindspore import dataset,context,nn
from mindspore.ops import functional as F
import mindspore.ops as ops
import random
import time
import numpy as np
from datetime import datetime
from the_net import Baseline
from data import GetDatasetGenerator,get_iterator,TestDataset
from tensorboardX import SummaryWriter
import logging
from options import opt
from models import *
import os


# build the model
model = Baseline()

# load checkpoint
if opt.load is not None:
    mindspore.load_checkpoint(opt.load, net=model)
    print('load model from', opt.load)

# GIE
nett = Net()
mindspore.load_param_into_net(nett, mindspore.load_checkpoint(opt.parameter))
nett.set_train(False)

optimizer = mindspore.nn.optim.Adam(model.trainable_params(), opt.lr)

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
t_root = opt.t_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_t_root = opt.test_t_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_iterator, iterations_epoch = get_iterator(opt.rgb_root, opt.gt_root, opt.t_root,\
         opt.batchsize,opt.trainsize)
test_loader = TestDataset(test_image_root, test_gt_root, test_t_root, opt.trainsize)
print(iterations_epoch)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Baseline-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};load:{};save_path:{};'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize,  opt.load, save_path,))

bce_loss = mindspore.nn.BCELoss(reduction='mean')


def bce_iou_loss(pred, mask):
    # weit = 1 + 5 * F.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = 1 + 5 * F.abs(F.avg_pool2d(mask, kernel_size=31, strides=1, pad_mode='same') - mask)
    # weights = ops.OnesLike(pred)
    # pos_weight = weights
    # wbce = F.binary_cross_entropy_with_logits(pred, mask, weights,reduction='none')
    # nnwbce = nn.BCEWithLogitsLoss(reduction='none')
    # nnwbce=bce_loss
    wbce = bce_loss(pred, mask)
    wbce = (weit * wbce).sum(axis =(2, 3)) / weit.sum(axis=(2, 3))
    sigmoid = ops.Sigmoid()
    pred = sigmoid(pred)
    inter = ((pred * mask) * weit).sum(axis=(2, 3))
    union = ((pred + mask) * weit).sum(axis=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# fixed random seed
def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_mindspore()

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

class ComputeLoss(nn.Cell):
    def __init__(self, network, loss_fn):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn

    def construct(self, images, label,  ts):
        R, L = nett(images)
        s1, ttt, u4, u3, u2, u1 = self.network(images, ts, L)
        loss_1 = self._loss_fn(u1, label)
        loss_2 = self._loss_fn(u2, label)
        loss_3= self._loss_fn(u3, label)
        loss_4 = self._loss_fn(u4, label)
        loss_5 = self._loss_fn(ttt, label)
        loss_6 = self._loss_fn(s1, label)
        # print(loss_1)
        return loss_1+loss_2+loss_3+loss_4+loss_5+loss_6

# train function
def train():
    # train
    net = ComputeLoss(model,bce_loss)
    T_net = nn.TrainOneStepCell(net, optimizer)
    epoch = opt.epoch
    print("==================Starting Training==================")
    for i in range(epoch):
        model.set_train(True)
        loss_all = 0
        epoch_num = i + 1
        if epoch_num==30:
            net = ComputeLoss(model, bce_iou_loss)
            T_net = nn.TrainOneStepCell(net, optimizer)
        if epoch_num==45:
            new_lr=1e-5
            ops.assign(optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))
            print(optimizer.learning_rate.data.asnumpy())
            T_net = nn.TrainOneStepCell(net, optimizer)
        if epoch_num==90:
            new_lr = 1e-6
            ops.assign(optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))
            print(optimizer.learning_rate.data.asnumpy())
            T_net = nn.TrainOneStepCell(net, optimizer)
        time_begin_epoch = time.time()
        for iteration, data in enumerate(train_iterator, start=1):
            data["rgb"]=F.squeeze(data["rgb"])
            data["gt"]=F.squeeze(data["gt"],axis=(1))
            data["t"]=F.squeeze(data["t"],axis=(1))
            loss_step = T_net(data["rgb"], data["gt"], data["t"])

            loss_all += loss_step.asnumpy()

            if iteration % 50 == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch_num, epoch, iteration, iterations_epoch, loss_step.asnumpy()))
        time_end_epoch = time.time()
        print('Epoch [{:03d}/{:03d}]:Loss_AVG={:.4f}, Time:{:.2f}'.
              format(epoch_num, epoch, loss_all / iterations_epoch, time_end_epoch - time_begin_epoch))
        if epoch_num % 5 == 0:
            mindspore.save_checkpoint(model, './TNet/TNet_epoch_{}_checkpoint.pth'.format(epoch_num))

        #test
        global best_mae, best_epoch
        mae_sum = 0
        model.set_train(False)
        for i in range(test_loader.size):
            image, gt, t, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)

            image = mindspore.Tensor(image)
            t = mindspore.Tensor(t)

            R, L = nett(image)
            res, ttt, u4, u3, u2, u1 = model(image, t, L)
            res = ops.interpolate(res, sizes=gt.shape, mode='bilinear', coordinate_transformation_mode="align_corners")
            res = res.squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res.asnumpy()
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch_num, mae, best_mae, best_epoch))
        if epoch_num == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch_num
                mindspore.save_checkpoint(model, './TNet/TNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch_num))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch_num, mae, best_epoch, best_mae))
    print("==================Ending Training==================")

if __name__ == '__main__':
    print("Start train...")
    from options import parser
    import warnings
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    # Train
    seed_mindspore()
    train()