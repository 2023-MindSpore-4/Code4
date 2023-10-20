# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import logging
import argparse
import datetime
import numpy as np
from time import time
import json

from pathlib import Path

from losses import DistillationLoss
import models
import math
import os
from models import deit_base_patch16_224, student_deit_base_patch16_224_6layer
import math
import os
from PIL import Image
from ipdb import set_trace as stxx
from mindcv.data import create_dataset, create_loader, create_transforms
from mindspore import Tensor, load_checkpoint,ops
import mindspore as ms
import ipdb
import mindspore as ms
from mindspore import SummaryRecord, Tensor, nn, ops
from mindspore.amp import StaticLossScaler
from mindspore.communication import get_group_size, get_rank, init
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from mindcv.data import create_dataset, create_loader, create_transforms
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import AllReduceSum, CheckpointManager, NoLossScaler
from mindcv.utils.random import set_seed

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
h1 = logging.StreamHandler()
formatter1 = logging.Formatter("%(message)s")
logger.addHandler(h1)
h1.setFormatter(formatter1)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
def train_epoch(
    teacher_model,
    student_model,
    dataset,
    loss_fn,
    optimizer,
    epoch,
    n_epochs,
    loss_scaler,
    reduce_fn=None,
    summary_record=None,
    rank_id=None,
    log_interval=100,
):
    """Training an epoch network"""

    # Define forward function
    def forward_fn(data, label):
        logits,_,_,_ = student_model(data,Tensor([1, 2, 3]), False)
        loss = loss_fn(logits, label)
        loss = loss_scaler.scale(loss)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    if args.distributed:
        mean = _get_gradients_mean()
        degree = _get_device_num()
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity

    # Define function of one-step training
#     @jit
    def train_step(data, label):

#         import mindspore as ms
        (loss, logits), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        status = ms.amp.all_finite(grads)
        if status:
            loss = loss_scaler.unscale(loss)
            grads = loss_scaler.unscale(grads)
            loss = ops.depend(loss, optimizer(grads))
        loss = ops.depend(loss, loss_scaler.adjust(status))
#         ipdb.set_trace()
        return loss, logits

    student_model.set_train()
    n_batches = dataset.get_dataset_size()
    n_steps = n_batches * n_epochs
    epoch_width, batch_width, step_width = len(str(n_epochs)), len(str(n_batches)), len(str(n_steps))  # noqa: F841
    total, correct = 0, 0

    start = time()

    num_batches = dataset.get_dataset_size()
#     ipdb.set_trace()
    for batch, (data,c_la, label) in enumerate(dataset.create_tuple_iterator()):
#     for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss, logits = train_step(data, label)

        if len(label.shape) == 1:
            correct += (logits.argmax(1) == label).asnumpy().sum()
        else:  # one-hot or soft label
            correct += (logits.argmax(1) == label.argmax(1)).asnumpy().sum()
        total += len(data)

        if (batch + 1) % log_interval == 0 or (batch + 1) >= num_batches or batch == 0:
            step = epoch * n_batches + batch
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor(step)).asnumpy()
            else:
                cur_lr = optimizer.learning_rate.asnumpy()
            logger.info(
                f"Epoch:[{epoch+1:{epoch_width}d}/{n_epochs:{epoch_width}d}], "
                f"batch:[{batch+1:{batch_width}d}/{n_batches:{batch_width}d}], "
                f"loss:{loss.asnumpy():8.6f}, lr: {cur_lr:.7f},  time:{time() - start:.6f}s"
            )
            start = time()
            if rank_id in [0, None]:
                if not isinstance(loss, Tensor):
                    loss = Tensor(loss)
                if summary_record is not None:
                    summary_record.add_value("scalar", "loss", loss)
                    summary_record.record(step)

    if args.distributed:
        correct = reduce_fn(Tensor(correct, ms.float32))
        total = reduce_fn(Tensor(total, ms.float32))
        correct /= total
        correct = correct.asnumpy()
    else:
        correct /= total

    if rank_id in [0, None]:
        logger.info(f"Training accuracy: {(100 * correct):0.3f}")
        if not isinstance(correct, Tensor):
            correct = Tensor(correct)
        if summary_record is not None:
            summary_record.add_value("scalar", "train_dataset_accuracy", correct)
            summary_record.record(step)

    return loss


def test_epoch(network, dataset, reduce_fn=None, rank_id=None):
    """Test network accuracy and loss."""
    network.set_train(False)  # TODO: check freeze

    correct, total = 0, 0
    for data, c_la, label in tqdm(dataset.create_tuple_iterator()):
        pred,_,_,_ = network(data)
        total += len(data)
        if len(label.shape) == 1:
            correct += (pred.argmax(1) == label).asnumpy().sum()
        else:  # one-hot or soft label
            correct += (pred.argmax(1) == label.argmax(1)).asnumpy().sum()

    if rank_id is not None:
        # dist_sum = AllReduceSum()
        correct = reduce_fn(Tensor(correct, ms.float32))
        total = reduce_fn(Tensor(total, ms.float32))
        correct /= total
        correct = correct.asnumpy()
    else:
        correct /= total

    return correct

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    
    parser.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--distributed', type=str2bool, nargs='?', const=True, default=False,
                       help='Run distribute (default=False)')    
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--student_model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
#     parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
#                         help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
#     parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                         help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--amp_level', type=str, default='O0',
                       help='Amp level - Auto Mixed Precision level for saving memory and acceleration. '
                            'Choice: O0 - all FP32, O1 - only cast ops in white-list to FP16, '
                            'O2 - cast all ops except for blacklist to FP16, '
                            'O3 - cast all ops to FP16. (default="O0").')    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Distillation parameters
    parser.add_argument('--teacher_model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher_path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_dir', default='/userhome/dataset/cifar-100-binary', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'IMNET','CIFAR_LT', 'CIFAR10_LT', 'INAT', 'INAT19'],
                        type=str, help='dataset path')
    
    parser.add_argument('--train_split', type=str, default='train',
                       help='Dataset train split name (default="train")')
    parser.add_argument('--val_split', type=str, default='test',
                       help='Dataset validation split name (default="val")')
    parser.add_argument('--dataset_download', type=str2bool, nargs='?', const=True, default=False,
                       help='If downloading the dataset, only support Mnist, Cifar10 and Cifar100 (default=False)')
    parser.add_argument('--num_parallel_workers', type=int, default=8,
                       help='Number of parallel workers (default=8)')
    parser.add_argument('--shuffle', type=str2bool, nargs='?', const=True, default=True,
                       help='Whether or not to perform shuffle on the dataset (default=True)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of elements to sample. None means sample all elements (default=None)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Number of batch size (default=128)')
    parser.add_argument('--drop_remainder', type=str2bool, nargs='?', const=True, default=True,
                       help='Determines whether or not to drop the last block whose data '
                            'row number is less than batch size (default=True)')

    parser.add_argument('--scale', type=tuple, default=(0.08, 1.0),
                       help='Random resize scale (default=(0.08, 1.0))')
    parser.add_argument('--ratio', type=tuple, default=(0.75, 1.333),
                       help='Random resize aspect ratio (default=(0.75, 1.333))')    
    parser.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability (default=0.5)')
    parser.add_argument('--vflip', type=float, default=0.0,
                       help='Vertical flip training aug probability (default=0.0)')
    parser.add_argument('--color_jitter', type=float, default=0.4,
                       help='Color jitter factor (default=0.4)')
    parser.add_argument('--interpolation', type=str, default='bilinear',
                       help='Image interpolation mode for resize operator(default="bilinear")')
    parser.add_argument('--auto_augment', type=str, default=None,
                       help='AutoAugment policy. "randaug" for RandAugment, "autoaug" for original AutoAugment, '
                            '"autoaugr" for AutoAugment with increasing posterize. and "3a" for AutoAugment with only 3'
                            'transformations. '
                            '"augmix" for AugmixAugment. '
                            '"trivialaugwide" for TrivialAugmentWide. '
                            'If apply, recommend for imagenet: randaug-m7-mstd0.5 (default=None).'
                            'Example: "randaug-m10-n2-w0-mstd0.5-mmax10-inc0", "autoaug-mstd0.5" or autoaugr-mstd0.5.')
    parser.add_argument('--aug_splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 3 (currently, only support 3 splits))'
                       'it should be set with one auto_augment')
    parser.add_argument('--re_prob', type=float, default=0.0,
                       help='Probability of performing erasing (default=0.0)')
    parser.add_argument('--re_scale', type=tuple, default=(0.02, 0.33),
                       help='Range of area scale of the erased area (default=(0.02, 0.33))')
    parser.add_argument('--re_ratio', type=tuple, default=(0.3, 3.3),
                       help='Range of aspect ratio of the erased area (default=(0.3, 3.3))')
    parser.add_argument('--re_value', default=0,
                       help='Pixel value used to pad the erased area (default=0),'
                            'please refer to mindspore.dataset.vision.RandomErasing.')
    parser.add_argument('--re_max_attempts', type=int, default=10,
                       help='The maximum number of attempts to propose a valid erased area, '
                            'beyond which the original image will be returned (default=10)')
    
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--mean', type=list, default=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.485 * 255, 0.456 * 255, 0.406 * 255])')
    parser.add_argument('--std', type=list, default=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                       help='List or tuple of mean values for each channel, '
                            'with respect to channel order (default=[0.229 * 255, 0.224 * 255, 0.225 * 255])')
    parser.add_argument('--crop_pct', type=float, default=0.875,
                       help='Input image center crop percent (default=0.875)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Hyperparameter of beta distribution of mixup. '
                            'Recommended value is 0.2 for ImageNet (default=0.0)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                       help='Hyperparameter of beta distribution of cutmix (default=0.0)')
    parser.add_argument('--cutmix_prob', type=float, default=1.0,
                       help='Probability of applying cutmix and/or mixup (default=1.0)')
    parser.add_argument('--aug_repeats', type=int, default=0,
                       help='Number of dataset repetition for repeated augmentation. '
                            'If 0 or 1, repeated augmentation is disabled. '
                            'Otherwise, repeated augmentation is enabled and the common choice is 3 (default=0)')
    
    parser.add_argument('--ckpt_save_interval', type=int, default=1,
                       help='Checkpoint saving interval. Unit: epoch (default=1)')
    parser.add_argument('--ckpt_save_policy', type=str, default='latest_k',
                       help='Checkpoint saving strategy. The optional values is '
                            'None, "top_k" or "latest_k" (default="latest_k")')
    # Optimize parameters
    parser = parser.add_argument_group('Optimizer parameters')
    parser.add_argument('--opt', type=str, default='adam',
                       choices=['sgd', 'momentum', 'adam', 'adamw', 'lion', 'rmsprop', 'adagrad', 'lamb', "nadam"],
                       help='Type of optimizer (default="adam")')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Hyperparameter of type float, means momentum for the moving average. '
                            'It must be at least 0.0 (default=0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay (default=1e-6)')
    parser.add_argument('--use_nesterov', type=str2bool, nargs='?', const=True, default=False,
                       help='Enables the Nesterov momentum (default=False)')
    parser.add_argument('--filter_bias_and_bn', type=str2bool, nargs='?', const=True, default=True,
                       help='Filter Bias and BatchNorm (default=True)')
    parser.add_argument('--eps', type=float, default=1e-10,
                       help='Term Added to the Denominator to Improve Numerical Stability (default=1e-10)')    
    
    
    # Loss parameters
    parser = parser.add_argument_group('Loss parameters')
    parser.add_argument('--loss', type=str, default='CE', choices=['BCE', 'CE'],
                       help='Type of loss, BCE (BinaryCrossEntropy) or CE (CrossEntropy)  (default="CE")')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Use label smoothing (default=0.0)')
    parser.add_argument('--aux_factor', type=float, default=0.0,
                       help='Aux loss factor (default=0.0)')
    parser.add_argument('--reduction', type=str, default='mean',
                       help='Type of reduction to be applied to loss (default="mean")')
    
    parser.add_argument('--scheduler', type=str, default='cosine_decay',
                       choices=['constant', 'cosine_decay', 'exponential_decay', 'step_decay',
                                'multi_step_decay', 'one_cycle', 'cyclic'],
                       help='Type of scheduler (default="cosine_decay")')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs (default=3)')
    parser.add_argument('--warmup_factor', type=float, default=0.0,
                       help='Warmup factor of learning rate (default=0.0)')
    parser.add_argument('--multi_step_decay_milestones', type=list, default=[30, 60, 90],
                       help='List of epoch milestones for lr decay, which is ONLY effective for '
                            'the multi_step_decay scheduler. LR will be decay by decay_rate at the milestone epoch.')
    parser.add_argument('--epoch_size', type=int, default=90,
                       help='Train epoch size (default=90)')
    
    parser.add_argument('--loss_scale', type=float, default=1.0,
                       help='Loss scale (default=1.0)')
    
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Interval for print training log. Unit: step (default=100)')
    parser.add_argument('--val_interval', type=int, default=100,
                       help='Interval for print training log. Unit: step (default=100)')
    return parser

def soft_cross_entropy(predicts, targets):
    student_likelihood = F.log_softmax(predicts, axis=-1)
    targets_prob = F.softmax(targets, axis=-1)
    return (-targets_prob * student_likelihood).mean()


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, axis=-1)
    _kl = ops.sum(p * (F.log_softmax(p_logit, axis=-1)
                         - F.log_softmax(q_logit, axis=-1)), 1)
    return ops.mean(_kl)


def main(args):
    print(args)
    # fix the seed for reproducibility
    seed = args.seed 
#     torch.manual_seed(seed)
    np.random.seed(seed)
#     random.seed(seed)

    if args.distributed:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )
        dist_sum = AllReduceSum()
    else:
        device_num = None
        rank_id = None
        dist_sum = None
    
 # create dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download,
        num_aug_repeats=args.aug_repeats,
    )
    train_count = dataset_train.get_dataset_size()
    num_classes = dataset_train.num_classes()



    # create transforms
    num_aug_splits = 0

    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=True,
        image_resize=args.input_size,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts,
        separate=num_aug_splits > 0,
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_prob=args.cutmix_prob,
        num_classes=num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
        separate=num_aug_splits > 0,
    )

    dataset_eval = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.val_split,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download,
    )

    transform_list_eval = create_transforms(
        dataset_name=args.dataset,
        is_training=False,
        image_resize=args.input_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
    )

    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        drop_remainder=False,
        is_training=False,
        transform=transform_list_eval,
        num_parallel_workers=args.num_parallel_workers,
    )

    print(f"Creating model: {args.teacher_model}")
    teacher_model = deit_base_patch16_224(pretrained=False,
                                                        num_classes=1000,
                                                        drop_rate=0.0,
                                                        drop_path_rate=0.1,)    
    print(f"Creating model: {args.student_model}")
    student_model = student_deit_base_patch16_224_6layer(pretrained=False,
                                                        num_classes=num_classes,
                                                        drop_rate=0.0,
                                                        drop_path_rate=0.1,)

    
#     teacher_model_dict =  load_checkpoint(args.teacher_path)
    # stxx()
#     ms.load_param_into_net(teacher_model, teacher_model_dict)
    

    # create loss
    ms.amp.auto_mixed_precision(student_model, amp_level=args.amp_level)
    loss = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )

    # create learning rate schedule
    num_batches = loader_train.get_dataset_size()
    lr_scheduler = create_scheduler(
        num_batches,
        scheduler=args.scheduler,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        milestones=args.multi_step_decay_milestones,
        num_epochs=args.epoch_size,
    )
    

    output_dir = Path(args.output_dir)


#     if args.eval:
#         test_stats = teacher_evaluate(data_loader_val, teacher_model)
#         print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
#         return

    # create optimizer
    optimizer = create_optimizer(
        student_model.trainable_params(),
        opt=args.opt,
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.use_nesterov,
        filter_bias_and_bn=args.filter_bias_and_bn,
        loss_scale=args.loss_scale,
        checkpoint_path='',
    )

    # set loss scale for mixed precision training
    if args.amp_level != "O0":
        loss_scaler = StaticLossScaler(args.loss_scale)
    else:
        loss_scaler = NoLossScaler()
        
    # log
    if rank_id in [None, 0]:
        logger.info("-" * 40)
        logger.info(
            f"Num devices: {device_num if device_num is not None else 1} \n"
            f"Distributed mode: {args.distributed} \n"
            f"Num training samples: {train_count}"
        )
        logger.info(
            f"Num classes: {num_classes} \n"
            f"Num batches: {num_batches} \n"
            f"Batch size: {args.batch_size} \n"
            f"Auto augment: {args.auto_augment} \n"
            f"Model: {args.student_model} \n"
#             f"Model param: {num_params} \n"
            f"Num epochs: {args.epoch_size} \n"
            f"Optimizer: {args.opt} \n"
            f"LR: {args.lr} \n"
            f"LR Scheduler: {args.scheduler}"
        )
        logger.info("-" * 40)

            
        logger.info("Start training")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        log_path = os.path.join(args.output_dir, "result.log")
        if not (os.path.exists(log_path) and args.output_dir != ""):  # if not resume training
            with open(log_path, "w") as fp:
                fp.write("Epoch\tTrainLoss\tValAcc\tTime\n")
                
    best_acc = 0
    summary_dir = f"./{args.output_dir}/summary_01"                
                
    manager = CheckpointManager(ckpt_save_policy=args.ckpt_save_policy)
    with SummaryRecord(summary_dir) as summary_record:
        for t in range(args.epoch_size):
            epoch_start = time()

            train_loss = train_epoch(
                teacher_model,
                student_model,
                loader_train,
                loss,
                optimizer,
                epoch=t,
                n_epochs=args.epoch_size,
                loss_scaler=loss_scaler,
                reduce_fn=dist_sum,
                summary_record=summary_record,
                rank_id=rank_id,
                log_interval=args.log_interval,
            )

            # val while train
            test_acc = Tensor(-1.0)
            
            args.val_while_train = True
            if args.val_while_train:
                if ((t + 1) % args.val_interval == 0) or (t + 1 == args.epoch_size):
                    if rank_id in [None, 0]:
                        logger.info("Validating...")
                    val_start = time()
                    test_acc = test_epoch(student_model, loader_eval, dist_sum, rank_id=rank_id)
                    test_acc = 100 * test_acc
                    if rank_id in [0, None]:
                        val_time = time() - val_start
                        logger.info(f"Val time: {val_time:.2f} \t Val acc: {test_acc:0.3f}")
                        if test_acc > best_acc:
                            best_acc = test_acc
                            save_best_path = os.path.join(args.output_dir, f"{args.model}-best.ckpt")
                            ms.save_checkpoint(student_model, save_best_path, async_save=True)
                            logger.info(f"=> New best val acc: {test_acc:0.3f}")

                        # add to summary
                        current_step = (t + 1) * num_batches + begin_step
                        if not isinstance(test_acc, Tensor):
                            test_acc = Tensor(test_acc)
                        if summary_record is not None:
                            summary_record.add_value("scalar", "test_dataset_accuracy", test_acc)
                            summary_record.record(int(current_step))

            # Save checkpoint
            if rank_id in [0, None]:
                if ((t + 1) % args.ckpt_save_interval == 0) or (t + 1 == args.epoch_size):
#                     if need_flush_from_cache:
#                         need_flush_from_cache = flush_from_cache(network)

                    ms.save_checkpoint(
                        optimizer, os.path.join(args.output_dir, f"{args.model}_optim.ckpt"), async_save=True
                    )
                    save_path = os.path.join(args.output_dir, f"{args.model}-{t + 1}_{num_batches}.ckpt")
                    ckpoint_filelist = manager.save_ckpoint(
                        student_model, num_ckpt=args.keep_checkpoint_max, metric=test_acc, save_path=save_path
                    )
                    if args.ckpt_save_policy == "top_k":
                        checkpoints_str = "Top K accuracy checkpoints: \n"
                        for ch in ckpoint_filelist:
                            checkpoints_str += "{}\n".format(ch)
                        logger.info(checkpoints_str)
                    else:
                        logger.info(f"Saving model to {save_path}")

                epoch_time = time() - epoch_start
                logger.info(f"Epoch {t + 1} time:{epoch_time:.3f}s")
                logger.info("-" * 80)
                with open(log_path, "a") as fp:
                    fp.write(f"{t+1}\t{train_loss.asnumpy():.7f}\t{test_acc.asnumpy():.3f}\t{epoch_time:.2f}\n")

    logger.info("Done!")                
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
