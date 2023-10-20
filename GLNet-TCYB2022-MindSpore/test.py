# %%

from dataset import *
from network import *

def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

os.environ['CUDA_VISIBLE_DEVICES']= "0"
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
seed_mindspore()
work_space = r'./'

output_folder = r'./Outputs'
checkpoints_folder = r'./Checkpoints'
dataset_root = os.path.join(work_space, 'Data')

test_set = 'Cosal2015'
#test_set = 'MSRC'
#test_set = 'iCoseg'
mean, std = dataset_mean_std(test_set)
ti = Compose([vision.ToTensor(), vision.Normalize(mean, std, is_hwc=False)])
# ti = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
is_shuffle = False
M = 5  
info_file = 'Info__' + test_set + '.txt'
list_file = 'List__' + test_set + '.txt'
GroupNumbers, GroupNames, GroupSizes, GroupFileLists = CoSOD_Dataset_Parser(dataset_root, info_file, list_file)
os.makedirs(os.path.join(output_folder, test_set), exist_ok=True)
for g in range(GroupNumbers):
    group_file_list = GroupFileLists[g]
    if is_shuffle:
        np.random.shuffle(group_file_list)
    gs = len(group_file_list)
    if np.mod(gs, M) != 0:
        num_add = M - np.mod(gs, M)
        for n in range(num_add):
            group_file_list.append(group_file_list[n])
        GroupSizes[g] += num_add


net = GLNet(return_loss=False)
resume_net_params = os.path.join(checkpoints_folder, 'trained', 'GLNet.ckpt')
# net.load_state_dict(torch.load(resume_net_params))
param_dict = ms.load_checkpoint(resume_net_params)
ms.load_param_into_net(net, param_dict)
model = ms.Model(net)
# model.set_train()
cat = P.Concat(axis=0)
unsqueeze = ops.ExpandDims()
# %%
for group_id in range(GroupNumbers):
    group_file_list = GroupFileLists[group_id]
    for index in range(0, len(group_file_list), M):
        images_bag = []
        names_bag = []
        for j in range(index, index + M):
            
            images_bag.append(unsqueeze(Tensor(ti(Image.open(os.path.join(dataset_root, group_file_list[j]) + '.jpg'))),0))
            names_bag.append(group_file_list[j].split('/')[-1])

            # image = Tensor(ti(rgb_loader(os.path.join(dataset_root, local_path+'.jpg'))))
            # label = Tensor(tl(binary_loader(os.path.join(dataset_root, local_path+'.png'))))
            # images_bag.append(ops.ExpandDims()(image,0))
            # labels_bag.append(ops.ExpandDims()(label,0))


        input_size = 160 
        grp_images = unsqueeze(cat(images_bag),0)
        grp_labels = Tensor(np.zeros((1, M, 1, input_size, input_size),
                                             dtype=np.float16), mstype.float16)

        cosod_maps = model.predict(grp_images)

        for m in range(M):
            PATH = os.path.join(output_folder, test_set, names_bag[m] + '.png')
            save_smap(cosod_maps[0, m, ...], PATH)
    print('Processing {}/{}'.format(group_id + 1, GroupNumbers))

# %%

#import os

#os.system('rm -rf ./Outputs/Cosal2015/*.png')
#os.system('rm -rf ./Outputs/iCoseg/*.png')
#os.system('rm -rf ./Outputs/MSRC/*.png')
