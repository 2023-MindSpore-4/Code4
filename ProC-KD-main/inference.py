import argparse
import datetime
import numpy as np
import time
import json
from tqdm import tqdm
from pathlib import Path
from models import student_deit_base_patch16_224_6layer
import math
import os
from PIL import Image
from ipdb import set_trace as stxx
from mindcv.data import create_dataset, create_loader, create_transforms
from mindspore import Tensor, load_checkpoint,ops
import mindspore as ms

def validate(model, dataset):
#     stxx()
    """Evaluates model on validation data with top-1 & top-5 metrics."""
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, acc1, acc5 = 0, 0, 0
    for data, _, label in tqdm(dataset.create_tuple_iterator(), total=num_batches):
        pred,pred_aug,_,_ = model(data,Tensor([1, 2, 3]), False)
        total += len(data)
#         test_loss += loss_fn(pred, label).asnumpy()
        acc1 += ops.intopk(pred, label, 1).sum().asnumpy()
        acc5 += ops.intopk(pred_aug, label, 5).sum().asnumpy()
    acc1 /= total
    acc5 /= total
    return acc1, acc5



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)

    # Model parameters
    parser.add_argument('--student_model', default='student_deit_base_patch16_224_6layer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--data_dir', default='/Volumes/lideng/datasets/cifar-100-binary', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100', 'IMNET', 'CIFAR_LT', 'CIFAR10_LT', 'INAT', 'INAT19', 'officehome'],
                        type=str, help=' dataset path')

    parser.add_argument('--checkpoint_path', default='./best_model_mindspore.ckpt', help='resume from checkpoint')

    return parser



parser = argparse.ArgumentParser('DeiT evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


print(f"Creating model: {args.student_model}")

nb_classes = 100

# create model
model = student_deit_base_patch16_224_6layer(pretrained=False,
                                                        num_classes=nb_classes,
                                                        drop_rate=0.0,
                                                        drop_path_rate=0.1,)
# load checkpoint
best_model_dict =  load_checkpoint(args.checkpoint_path)
# for k, v in best_model_dict.items():
#     print(k)
# stxx()
ms.load_param_into_net(model, best_model_dict)


dataset_eval = create_dataset(
    name=args.dataset,
    root=args.data_dir,
    split='test',
    num_shards=None,
    shard_id=None,
    num_parallel_workers=4,
    download=False,
)

transform_list_eval = create_transforms(
    dataset_name=args.dataset,
    is_training=False,
    image_resize=args.input_size,
    crop_pct=0.875,
    interpolation="bicubic",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.120000000000005, 57.375],
)

loader_eval = create_loader(
    dataset=dataset_eval,
    batch_size=args.batch_size,
    drop_remainder=False,
    is_training=False,
    transform=transform_list_eval,
    num_parallel_workers=4,
)

# validate
print("Testing...")
test_acc1, test_acc5 = validate(model, loader_eval)
print(f"Acc@1: {test_acc1:.4%}, Acc@5: {test_acc5:.4%}")
