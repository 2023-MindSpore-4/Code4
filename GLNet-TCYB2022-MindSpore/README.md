# GLNet-TCYB2022-MindSpore
* Pretrained model:
  - We provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the checkpoint `model_init.ckpt` to `Checkpoints/warehouse/` folder 
  and put the pretrained backbone `backbone.ckpt` to `Checkpoints/warehouse/` folder.
  - Pretrained model download:
```
Baidu Cloud: https://pan.baidu.com/s/1qHYtJtSFD65asNOdBnxTdQ?pwd=lhqs  Password: lhqs 
```

# MindSpore
* MindSpore implementation of GLNet

## Requirements

* mindspore-gpu 1.9.0
* python 3.8
* GPU cuda 11.1

## Data Preprocessing
* We resize the images of original test datasets. Please download the resized data, and put the data to `Data/` folder.
* Resized test datasets:
```
Baidu Cloud: https://pan.baidu.com/s/1sXBc4H3fKK8Y8ceaU4AjSQ  Password: 0224
```

## Test
```
python test.py
```

* You can find the results in the `'Outputs/'` folder.

# Bibtex
If you use the results and code, please cite our paper.
```
@article{GLNet,
  title={Global-and-local collaborative learning for co-Salient object detection},
  author={Cong, Runmin and Yang, Ning and Li, Chongyi and Fu, Huazhu and Zhao, Yao and Huang, Qingming and Kwong, Sam},
  journal={IEEE Transactions on Cybernetics},
  year={2022},
  publisher={IEEE}
}
```
