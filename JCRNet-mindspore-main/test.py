"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset import GeneratorDataset
from mindspore.ops import functional as F

from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor, load_checkpoint

import utils

from data_RGB import get_test_data
from DGUNet_LBP9_M import DGUNet

from skimage import img_as_ubyte
import ipdb
from ipdb import set_trace as stxx
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='../Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/DGUNet/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/Deraining/models/DGUNet_LBP9_M/model_best_mindspore2.ckpt', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = DGUNet()

# utils.load_checkpoint(model_restoration,args.weights)

best_model_dict =  load_checkpoint(args.weights)
# for k, v in best_model_dict.items():
#     stxx()
#     print(k)
#     print('value:',v)   
ms.load_param_into_net(model_restoration, best_model_dict)

print("===>Testing using weights: ",args.weights)
# model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
# model_restoration.eval()

# datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']
datasets = ['SOTS']

def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def test_x8(model, L):
    E_list = [model(augment_img_tensor(L, mode=i))[0] for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, 'input')
    rgb_dir_test_target = os.path.join(args.input_dir, 'target')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader  = GeneratorDataset(source=test_dataset,column_names=["inp","filename"],shuffle=False, num_parallel_workers=4)

    result_dir  = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    for ii, data_test in enumerate(tqdm(test_loader), 0):

        input_    = data_test[0]
        filenames = data_test[1]
        restored = model_restoration(input_)
        restored = ops.clamp(restored[0], 0, 1)

        restored = restored.permute(0, 2, 3, 1).asnumpy()


        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, str(filenames)+'.png')), restored_img)


    noisy_images = []
    clean_images = []

    for filename in os.listdir(result_dir):
        if filename.endswith('png'):
            noisy_images.append(cv2.imread(os.path.join(result_dir,filename)))
            clean_images.append(cv2.imread(os.path.join(rgb_dir_test_target,filename)))

    # 计算PSNR和SSIM值
    psnr_values = []
    ssim_values = []

    for i in range(len(noisy_images)):
        ssim = structural_similarity(noisy_images[i], clean_images[i], multichannel=True)
        psnr = peak_signal_noise_ratio(noisy_images[i], clean_images[i])

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # 计算平均PSNR和SSIM值
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    print("平均PSNR值：", mean_psnr)
    print("平均SSIM值：", mean_ssim)