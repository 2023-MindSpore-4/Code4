from utils import *

from mindspore import dataset
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose

def dataset_mean_std(dataset):
    assert dataset in ['Cosal2015', 'CoSOD3k', 'iCoseg', 'MSRC']
    if dataset == 'Cosal2015':
        mean, std = [0.4898, 0.4624, 0.3941], [0.2175, 0.2139, 0.2109]
    if dataset == 'CoSOD3k':
        mean, std = [0.4883, 0.4642, 0.4083], [0.2180, 0.2138, 0.2135]
    if dataset == 'iCoseg':
        mean, std = [0.5129, 0.5223, 0.4806], [0.2080, 0.2021, 0.2025]
    if dataset == 'MSRC':
        mean, std = [0.5017, 0.5012, 0.4333], [0.2248, 0.2182, 0.2371]
    return mean, std

def CoSOD_Dataset_Parser(dataset_root, info_file, list_file):
    with open(os.path.join(dataset_root, info_file), 'r') as f:
        contents = f.read().splitlines()
    GroupNumbers = len(contents)
    GroupNames, GroupSizes = [], []
    for c in contents:
        g_name, g_size = c.split(' ')
        g_size = int(g_size)
        GroupNames.append(g_name)
        GroupSizes.append(g_size)   
    with open(os.path.join(dataset_root, list_file), 'r') as f:
        local_paths = f.read().splitlines()
    S = []
    for i in range(GroupNumbers):
        s = 0
        for j in range(i+1):
            s += GroupSizes[j]
        S.append(s)
    GroupFileLists = []
    for sid in range(GroupNumbers):
        if sid == 0:
            start_index = 0
            end_index = S[0]
        else:
            start_index = S[sid-1]
            end_index = S[sid]
        gr_file_list = []
        for index in range(start_index, end_index):
            gr_file_list.append(local_paths[index])
        GroupFileLists.append(gr_file_list)
    return GroupNumbers, GroupNames, GroupSizes, GroupFileLists


def Random_Batch_Group_Loader(dataset_root, GroupNumbers, GroupFileLists, B, M, dataset_mean, dataset_std):
    ti = Compose([vision.ToTensor(), vision.Normalize(dataset_mean, dataset_std, is_hwc=False)])
    tl = vision.ToTensor()
    cat = P.Concat(axis=0)
    group_ids = random.choice(GroupNumbers, B, replace=False)
    batch_group_images_bag = []
    batch_group_labels_bag = []
    batch_group_names_bag = []
    for g in group_ids:
        group_file_list = GroupFileLists[g]
        image_ids = random.choice(len(group_file_list), M, replace=False)
        images_bag = []
        labels_bag = []
        names_bag = []
        for i in image_ids:
            local_path = group_file_list[i]
            names_bag.append(local_path)
            image = ti(rgb_loader(os.path.join(dataset_root, local_path+'.jpg')))
            image = Tensor(ti(rgb_loader(os.path.join(dataset_root, local_path+'.jpg'))))
            label = Tensor(tl(binary_loader(os.path.join(dataset_root, local_path+'.png'))))
            images_bag.append(ops.ExpandDims()(image,0))
            labels_bag.append(ops.ExpandDims()(label,0))
        batch_group_images_bag.append(ops.ExpandDims()(cat(images_bag),0))
        batch_group_labels_bag.append(ops.ExpandDims()(cat(labels_bag),0))
        batch_group_names_bag.append(names_bag)
    batch_group_images = cat(batch_group_images_bag) 
    batch_group_labels = cat(batch_group_labels_bag)
    return batch_group_images, batch_group_labels, batch_group_names_bag

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')