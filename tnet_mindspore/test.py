import time
import mindspore
import mindspore as ms
from mindspore import dataset,context,nn
from mindspore.dataset import transforms,vision
import numpy as np
import mindspore.ops as ops
import sys
import numpy as np
import os, argparse
import cv2
from the_net import Baseline
from data import TestDataset
import numpy as np
import cv2
from models import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--parameter', default='./GIE.ckpt', help='name of parameter file')
parser.add_argument("--device_id", type=str, default='1', help="Device id")
parser.add_argument('--device_target', type=str, default="GPU",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_path',type=str,default='Dataset/VT5000/VT5000_clear/',help='test dataset path')
# parser.add_argument('--test_path',type=str,default='Dataset/VT1000/VT1000/',help='test dataset path')
# parser.add_argument('--test_path', type=str, default='Dataset/VT821/VT821/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
context.set_context(mode=context.PYNATIVE_MODE, device_target=opt.device_target)

# load the model
model = Baseline()

mindspore.load_param_into_net(model, mindspore.load_checkpoint('./TNet/TNet_epoch_best.pth.ckpt'))
model.set_train(False)

# GIE
net = Net()
mindspore.load_param_into_net(net, mindspore.load_checkpoint(opt.parameter))
net.set_train(False)

# test
# for 821,1000
# test_datasets = ['']
# for 5000
test_datasets = ['Test']

for dataset in test_datasets:
    save_path = 'Results/5000/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    t_root = dataset_path + dataset + '/T/'
    test_loader = TestDataset(image_root, gt_root, t_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, t, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image=mindspore.Tensor(image)
        t = mindspore.Tensor(t)
        R, L = net(image)

        res, ttt, u4, u3, u2, u1 = model(image, t, L)

        res = ops.interpolate(res, sizes=gt.shape, mode='bilinear', coordinate_transformation_mode="align_corners")
        res = res.squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res.asnumpy()
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, res * 255)
    print('Test Done!')