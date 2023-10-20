import logging
import math
import os
from typing import Any
import pickle
import random
import numpy as np
from PIL import Image
import mindspore
import mindspore.dataset as ds
from mindspore.dataset import transforms,vision
from mindspore import Tensor
# from mindspore.dataset import UnionBaseDataset, MappableDataset
# import torch.utils.data as data
# from torchvision import datasets
# from torchvision import transforms
from .mydataset import ImageFolder,ImageFolder_fix
from .randaugment import RandAugmentMC, MyRandAugmentMC




logger = logging.getLogger(__name__)

__all__ = ['TransformOpenMatch', 'TransformFixMatch', 'cifar10_mean',
           'cifar10_std', 'cifar100_mean', 'cifar100_std', 'normal_mean',
           'normal_std', 'TransformFixMatch_Imagenet', 'TransformTest', 'AdvDataset', 'TransformAug',
           'TransformFixMatch_Imagenet_Weak']
### Enter Path of the data directory.
DATA_PATH = './data'

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar(args, norm=True):
    root = args.root
    name = args.dataset
    if name == "cifar10":
        data_folder = CIFAR10FIX
        data_folder_main = CIFAR10SSL
        mean = cifar10_mean
        std = cifar10_std
        num_class = 10
    elif name == "cifar100":
        data_folder = CIFAR100FIX
        data_folder_main = CIFAR100SSL
        mean = cifar100_mean
        std = cifar100_std
        num_class = 100
        num_super = args.num_super

    else:
        raise NotImplementedError()
    assert num_class > args.num_classes

    if name == "cifar10":
        base_dataset = data_folder(root, usage='train')
        args.num_classes = 6
    elif name == 'cifar100':
        base_dataset = data_folder(root, usage='train',
                                    num_super=num_super)
        args.num_classes = base_dataset.num_known_class

    base_dataset.targets = np.array(base_dataset.targets)
    if name == 'cifar10':
        base_dataset.targets -= 2
        base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
        base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = \
        x_u_split(args, base_dataset.targets)

    ## This function will be overwritten in trainer.py
    # norm_func = TransformFixMatch(mean=mean, std=std, norm=norm)
    norm_func = TransformHOOD(mean=mean, std=std)

    if norm:
        norm_func_test = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)
        ])
    else:
        norm_func_test = transforms.Compose([
            vision.ToTensor(),
        ])

    if name == 'cifar10':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, usage='train',
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, usage='train',
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, usage='train',
            transform=norm_func_test)
    elif name == 'cifar100':
        train_labeled_dataset = data_folder_main(
            root, train_labeled_idxs, num_super = num_super, usage='train',
            transform=norm_func)
        train_unlabeled_dataset = data_folder_main(
            root, train_unlabeled_idxs, num_super = num_super, usage='train',
            transform=norm_func, return_idx=False)
        val_dataset = data_folder_main(
            root, val_idxs, num_super = num_super,usage='train',
            transform=norm_func_test)

    if name == 'cifar10':
        train_labeled_dataset.targets -= 2
        train_unlabeled_dataset.targets -= 2
        val_dataset.targets -= 2


    if name == 'cifar10':
        test_dataset = data_folder(
            root, usage='test', transform=norm_func_test)
    elif name == 'cifar100':
        test_dataset = data_folder(
            root, usage='test', transform=norm_func_test,num_super=num_super)
    test_dataset.targets = np.array(test_dataset.targets)

    if name == 'cifar10':
        test_dataset.targets -= 2
        test_dataset.targets[np.where(test_dataset.targets == -2)[0]] = 8
        test_dataset.targets[np.where(test_dataset.targets == -1)[0]] = 9

    target_ind = np.where(test_dataset.targets >= args.num_classes)[0]
    test_dataset.targets[target_ind] = args.num_classes


    unique_labeled = np.unique(train_labeled_idxs)
    val_labeled = np.unique(val_idxs)
    logger.info("Dataset: %s"%name)
    logger.info(f"Labeled examples: {len(unique_labeled)}"
                f"Unlabeled examples: {len(train_unlabeled_idxs)}"
                f"Valdation samples: {len(val_labeled)}")
    return train_labeled_dataset, train_unlabeled_dataset, \
           test_dataset, val_dataset



def get_imagenet(args, norm=True):
    mean = normal_mean
    std = normal_std
    txt_labeled = "filelist/imagenet_train_labeled.txt"
    txt_unlabeled = "filelist/imagenet_train_unlabeled.txt"
    txt_val = "filelist/imagenet_val.txt"
    txt_test = "filelist/imagenet_test.txt"
    ## This function will be overwritten in trainer.py
    norm_func = TransformFixMatch_Imagenet(mean=mean, std=std,
                                           norm=norm, size_image=224)
    dataset_labeled = ImageFolder(txt_labeled, transform=norm_func)
    dataset_unlabeled = ImageFolder_fix(txt_unlabeled, transform=norm_func)

    test_transform = transforms.Compose([
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.ToTensor(),
        vision.Normalize(mean=mean, std=std)
    ])
    dataset_val = ImageFolder(txt_val, transform=test_transform)
    dataset_test = ImageFolder(txt_test, transform=test_transform)
    logger.info(f"Labeled examples: {len(dataset_labeled)}"
                f"Unlabeled examples: {len(dataset_unlabeled)}"
                f"Valdation samples: {len(dataset_val)}")
    return dataset_labeled, dataset_unlabeled, dataset_test, dataset_val


def x_u_split(args, labels):
    label_per_class = args.num_labeled # // args.num_classes
    val_per_class = args.num_val // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    #if not args.no_out:
    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            vision.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong

class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT)])
        self.weak2 = transforms.Compose([
            vision.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std,is_hwc=False)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, strong


class TransformHOOD(object):
    def __init__(self, mean, std, size_image=32, aug_num=3):
        
        self.weak = transforms.Compose([
            vision.Resize((size_image, size_image)),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std,is_hwc=False)])

        self.pre_strong = transforms.Compose([
            vision.Resize((size_image, size_image)),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT),])
        self.ops = MyRandAugmentMC(m=10)
        self.post_strong = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std,is_hwc=False)])
        self.aug_num = aug_num

    def __call__(self, x):
        idx = random.randint(0, self.aug_num-1)
        weak = self.weak(x)
        strong = self.pre_strong(x)
        strong = self.ops(strong, idx)
        strong = self.post_strong(strong)
        return weak, strong, idx
        



class TransformFixMatch_Imagenet(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT)])
        self.weak2 = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong



class TransformFixMatch_Imagenet_Weak(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT)])
        self.weak2 = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            vision.Resize((256, 256)),
            vision.RandomHorizontalFlip(),
            vision.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode=vision.Border.REFLECT),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong



class TransformTest(object):
    def __init__(self, mean, std):
        self.transform = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        x_test = self.transform(x)
        return x_test



class TransformAug(object):
    def __init__(self, mean, std):
        self.transform = transforms.Compose([
            vision.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        x_test = self.transform(x)
        return x_test
    

class AdvDataset(ds.Dataset):
    # adv_data.shape = B X H X W X C
    # Image tensor data get normailized already
    def __init__(self, adv_data, labels, transform=None):
        self.data = adv_data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)

class CIFAR10FIX():
    def __init__(self, dataset_dir='./data/cifar-10-binary/cifar-10-batches-bin',
                 usage='train', transform=None,
                 target_transform=None, return_idx=False):
        self.transform = transform
        self.target_transform = target_transform

        self.data_iter = ds.Cifar10Dataset(dataset_dir=dataset_dir, usage=usage).create_dict_iterator()
        self.data = []
        self.targets = []

        for i, data in enumerate(self.data_iter):
            self.data.append(data["image"].asnumpy())
            self.targets.append(data["label"].asnumpy())

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # self.data=Tensor.from_numpy(self.data)
        # self.targets = Tensor.from_numpy(self.targets)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class CIFAR10SSL():
    def __init__(self, dataset_dir, indexs=None,usage='train',
                 transform=None, target_transform=None,
                return_idx=False):
        self.transform=transform
        self.target_transform=target_transform
        if usage=='train':self.shuffle=True
        else:self.shuffle=False

        self.data_iter=ds.Cifar10Dataset(dataset_dir=dataset_dir, usage=usage,shuffle=self.shuffle).create_dict_iterator()
        self.data=[]
        self.targets=[]

        for i, data in enumerate(self.data_iter):
            self.data.append(data["image"].asnumpy())
            self.targets.append(data["label"].asnumpy())

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # self.data=Tensor.from_numpy(self.data)
        # self.targets = Tensor.from_numpy(self.targets)

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        image, label = self.data_index[index], self.targets_index[index]
        image = Image.fromarray(image)
        # image=Tensor(image,mindspore.float32)
        # label=Tensor(label,mindspore.int32)

        # if self.transform is not None:
        #     image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if not self.return_idx:
            return image, label
        else:
            return image, label, index

    def __len__(self):
        return len(self.data_index)






class CIFAR100FIX():
    def __init__(self, dataset_dir, num_super=10, usage='train', transform=None,
                 target_transform=None,return_idx=False):
        self.transform=transform
        self.target_transform=target_transform
        if usage=='train':self.shuffle=True
        else:self.shuffle=False
        self.data_iter = ds.Cifar100Dataset(dataset_dir=dataset_dir, usage=usage,shuffle=self.shuffle,num_shards=num_super).create_dict_iterator()
        self.data = []
        self.targets = []
        for i, data in enumerate(self.data_iter):
            self.data.append(data["image"].asnumpy())
            self.targets.append(data["label"].asnumpy())

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.course_labels = coarse_labels[self.targets]
        self.targets = np.array(self.targets)
        labels_unknown = self.targets[np.where(self.course_labels > num_super)[0]]
        labels_known = self.targets[np.where(self.course_labels <= num_super)[0]]
        unknown_categories = np.unique(labels_unknown)
        known_categories = np.unique(labels_known)

        num_unknown = len(unknown_categories)
        num_known = len(known_categories)
        print("number of unknown categories %s"%num_unknown)
        print("number of known categories %s"%num_known)
        assert num_known + num_unknown == 100
        #new_category_labels = list(range(num_known))
        self.targets_new = np.zeros_like(self.targets)
        for i, known in enumerate(known_categories):
            ind_known = np.where(self.targets==known)[0]
            self.targets_new[ind_known] = i
        for i, unknown in enumerate(unknown_categories):
            ind_unknown = np.where(self.targets == unknown)[0]
            self.targets_new[ind_unknown] = num_known

        self.targets = self.targets_new
        assert len(np.where(self.targets == num_known)[0]) == len(labels_unknown)
        assert len(np.where(self.targets < num_known)[0]) == len(labels_known)
        self.num_known_class = num_known


    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class CIFAR100SSL(CIFAR100FIX):
    def __init__(self, dataset_dir, indexs, num_super=10, usage='train',
                 transform=None, target_transform=None,return_idx=False):
        super().__init__(dataset_dir=dataset_dir, num_super=num_super,usage=usage,
                         transform=transform,
                         target_transform=target_transform)
        self.return_idx = return_idx
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.set_index()
        
    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets


    def __getitem__(self, index):
        image, label = self.data_index[index], self.targets_index[index]
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        if not self.return_idx:
            return image, label
        else:
            return image, label, index

    def __len__(self):
        return len(self.data_index)

def get_transform(mean, std, image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            vision.Resize((image_size[0], image_size[1])),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            vision.Resize((image_size[0], image_size[1])),
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std),
        ])
        test_transform = vision.ToTensor()

    return train_transform, test_transform


def get_ood(dataset, id, test_only=False, image_size=None):
    image_size = (32, 32, 3) if image_size is None else image_size
    if id == "cifar10":
        mean = cifar10_mean
        std = cifar10_std
    elif id == "cifar100":
        mean = cifar100_mean
        std = cifar100_std
    elif "imagenet"  in id or id == "tiny":
        mean = normal_mean
        std = normal_std

    _, test_transform = get_transform(mean, std, image_size=image_size)

    if dataset == 'cifar10':
        test_set = ds.Cifar10Dataset('./datacifar-10-binary/cifar-10-batches-bin', usage='test')
        test_set=test_set.map(operations=test_transform,input_columns='image')

    elif dataset == 'cifar100':
        test_set = ds.Cifar100Dataset('./data/cifar-100-binary/cifar-100-binary', usage='test')
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'svhn':
        test_set = ds.SVHNDataset('./data/svhn',usage='test')
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')
    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'cub')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'flowers102':
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'food_101':
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'dtd':
        test_dir = os.path.join(DATA_PATH, 'dtd')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    elif dataset == 'pets':
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = ds.ImageFolderDataset(test_dir)
        test_set = test_set.map(operations=test_transform, input_columns='image')

    return test_set

DATASET_GETTERS = {'cifar10': get_cifar,
                   'cifar100': get_cifar,
                   'imagenet': get_imagenet,
                   }







