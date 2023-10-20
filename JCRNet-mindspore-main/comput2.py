import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# 读取噪声图和干净图文件夹中的所有图像
noisy_path = "results/DGUNet_Filter/SOTS/"
clean_path = "Datasets/test/target/"

noisy_images = []
clean_images = []

for filename in os.listdir(noisy_path):
    noisy_images.append(cv2.imread(os.path.join(noisy_path,filename)))
    clean_images.append(cv2.imread(os.path.join(clean_path,filename)))

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
