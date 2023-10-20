import os
import math
import random
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn, train,ops
from mindspore.common.initializer import Normal
from mindspore.dataset import DistributedSampler
from mindspore.dataset import RandomSampler,SequentialSampler
from .myDataLoader import DataLoader
# from mindspore import load_checkpoint, load_param_into_net
# import torch
# from torch import nn
#
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
# import torch.optim as optim
# from torch.optim.lr_scheduler import LambdaLR

from dataset.cifar import DATASET_GETTERS, get_ood

__all__ = ['create_model', 'set_model_config',
           'set_dataset', 'set_models',
           'save_checkpoint', 'set_seed']


def create_model(args):
    if 'wideresnet' in args.arch:
        import models.wideresnet as models
        model_c = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.num_classes,
                                        open=True)
        model_s = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0,
                                        num_classes=args.aug_num,
                                        open=True)
    elif args.arch == 'resnext':
        import models.resnext as models
        model_c = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
        model_s = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.aug_num)
    elif args.arch == 'resnet_imagenet':
        import models.resnet_imagenet as models
        model_c = models.resnet18(num_classes=args.num_classes)
        model_s = models.resnet18(num_classes=args.aug_num)

    return model_c, model_s



def set_model_config(args):

    if args.dataset == 'cifar10':
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 55
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'wideresnet_10':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == "imagenet":
        args.num_classes = 20

    args.image_size = (32, 32, 3)
    if args.dataset == 'cifar10':
        args.ood_data = ["svhn", 'cifar100']#, 'lsun', 'imagenet'

    elif args.dataset == 'cifar100':
        args.ood_data = ['cifar10', "svhn"]#, 'lsun', 'imagenet'

    elif 'imagenet' in args.dataset:
        args.ood_data = ['lsun', 'dtd', 'cub', 'flowers102',
                         'caltech_256', 'stanford_dogs']
        args.image_size = (224, 224, 3)

def set_dataset(args):
    labeled_dataset, unlabeled_dataset, test_dataset, val_dataset = \
        DATASET_GETTERS[args.dataset](args)

    ood_loaders = {}
    for ood in args.ood_data:
        print('OOD dataset: ', ood)
        ood_dataset = get_ood(ood, args.dataset, image_size=args.image_size)
        # ood_dataset=ood_dataset.batch(batch_size=args.batch_size,num_parallel_workers=args.num_workers)
        # ood_loaders[ood]=ood_dataset.create_dict_iterator()
        ood_loaders[ood] = DataLoader(ood_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    if args.local_rank == 0:
        ms.context.set_auto_parallel_context()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # labeled_data=labeled_dataset.batch(batch_size=args.batch_size,num_parallel_workers=args.num_workers,drop_remainder=True)
    # labeled_trainloader=labeled_data.create_dict_iterator()
    labeled_trainloader = DataLoader(
        labeled_dataset,
        # sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    # test_data=test_dataset.batch(batch_size=args.batch_size,num_parallel_workers=args.num_workers)
    # test_loader=test_data.create_dict_iterator()

    test_loader = DataLoader(
        test_dataset,
        # sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    # val_dataset=val_dataset.batch(batch_size=args.batch_size,num_parallel_workers=args.num_workers)
    # val_loader=val_dataset.create_dict_iterator()

    val_loader = DataLoader(
        val_dataset,
        # sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        ms.context.set_auto_parallel_context()

    return labeled_trainloader, unlabeled_dataset, \
           test_loader, val_loader , ood_loaders


# def get_cosine_schedule_with_warmup(optimizer,
#                                     num_warmup_steps,
#                                     num_training_steps,
#                                     num_cycles=7./16.,
#                                     last_epoch=-1):
#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / \
#             float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress))
#
#     return LambdaLR(optimizer, _lr_lambda, last_epoch)


def set_models(args):
    models = create_model(args)
    if args.local_rank == 0:
        ms.context.set_auto_parallel_context()
    for model in models:
        ms.context.set_auto_parallel_context()
        
    model_c, model_s = models

    no_decay = ['bias', 'bn']
    grouped_parameters_c = [
        {'params': model_c.trainable_params(), 'weight_decay': args.wdecay}
    ]
    
    grouped_parameters_s = [
        {'params': model_s.trainable_params(), 'weight_decay': args.wdecay}
    ]

    if args.opt == 'sgd':
        optimizer_c = nn.SGD(grouped_parameters_c, learning_rate=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        optimizer_s = nn.SGD(grouped_parameters_s, learning_rate=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
    elif args.opt == 'adam':
        optimizer_c = nn.Adam(grouped_parameters_c, learning_rate=2e-3)
        optimizer_s = nn.Adam(grouped_parameters_c, learning_rate=2e-3)

    # args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler_c = nn.cosine_decay_lr(min_lr=0.0,max_lr=0.1,total_step=args.total_steps,step_per_epoch=10,decay_epoch=args.warmup)
    scheduler_s = nn.cosine_decay_lr(min_lr=0.0,max_lr=0.1,total_step=args.total_steps,step_per_epoch=10,decay_epoch=args.warmup)

    return models, (optimizer_c, optimizer_s), (scheduler_c, scheduler_s)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    ms.save_checkpoint(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)
    if args.n_gpu > 0:
        ms.set_seed(args.seed)
