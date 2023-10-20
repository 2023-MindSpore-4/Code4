# TNet-MindSpore_TMM2022
Runmin Cong, Kepu Zhang, Chen Zhang, Feng Zheng, Yao Zhao, Qingming Huang, and Sam Kwong, "Does Thermal Really Always Matter for RGB-T Salient Object Detection?" IEEE Transactions on Multimedia, 2022.

# Results of TNet-MindSpore:
* Results:
  - We provide the resutls of our TNet-MindSpore on VT5000, VT1000, VT821 datasets. 
    ```
    Baidu Cloud: https://pan.baidu.com/s/1o7qjeDMwvcvTBhAomiQIKA    Password: 93tw 
    ```

# MindSpore Code of TNet:
* MindSpore-GPU 1.9.0 implementation of TNet
* Pretrained model:
  - We provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the `TNet_epoch_best.pth.ckpt` to `TNet/` folder.
  - Pretrained model download:
    ```
    Baidu Cloud: https://pan.baidu.com/s/1G6P05b2P1cguGO9nbzIvEw   Password: ye3w 
    ```


## Requirements

* python 3.8
* mindspore-gpu 1.9.0
* GPU cuda 11.1


## Data Preprocessing
* Please download the test data, unzip it, and put the `VT821`, `VT1000`, `VT5000` to `Dataset/` folder.
* train and test datasets:
```
Baidu Cloud: https://pan.baidu.com/s/1mpMKWf-fiN-oqQTepfoDzA   Password: nb9w
```

## Test
```
python test.py
```

* You can find the results in the `'Results/'` folder.


# If you use our TNet, please cite our paper:

@article{TNet,
  title={Does Thermal Really Always Matter for {RGB-T} Salient Object Detection?},
  author={Cong, Runmin and Zhang, Kepu and Zhang, Chen and Zheng, Feng and Zhao, Yao and Huang, Qingming and Kwong, Sam },
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}

# Contact Us:
If you have any questions, please contact Runmin Cong (rmcong@bjtu.edu.cn).
