import mindspore
from Network.snet import Snet
from data import dataset
import os
import cv2
import sys
from mindspore.dataset import GeneratorDataset
import numpy as np

TAG = "WSLNet"
SAVE_PATH = TAG
GPU_ID=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

DATASETS = ['./data_test/THUR', './data_test/PASCAL-S', './data_test/ECSSD', './data_test/HKU-IS', './data_test/DUTS-TE']
mindspore.set_context(device_target='GPU', device_id=0)

class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s" % self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, snapshot=sys.argv[1], mode='test')
        self.data = Dataset.Data(self.cfg)

        self.loader = GeneratorDataset(self.data, shuffle=True, num_parallel_workers=8).batch(1)
        ## network
        self.net = Network(self.cfg)

    def save(self):
        with mindspore.ops.stop_gradient():
            for image, mask, _, (H, W), name in self.loader:
                out = self.net(image.cuda().float())
                out = mindspore.ops.interpolate(out, sizes=(H, W), mode='bilinear')
                pred = (mindspore.ops.Sigmoid(out[0, 0]) * 255).cpu().numpy()
                head = './pred_maps/{}/'.format(TAG) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], np.uint8(pred))
if __name__=='__main__':
    for e in DATASETS:
        t =Test(dataset, e, Snet)
        t.save()