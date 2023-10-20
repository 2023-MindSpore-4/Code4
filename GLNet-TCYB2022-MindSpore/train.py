#%%

from dataset import *
from network import *
ms.set_context(mode=ms.PYNATIVE_MODE)
def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# gpu_devices = list(np.arange(torch.cuda.device_count()))
work_space = r'./'

multi_gpu = False
ms.set_context(device_target="GPU")
seed_mindspore()
output_folder = r'./Outputs'
checkpoints_folder = r'./Checkpoints'
dataset_root = os.path.join(work_space, 'Data')

#%%

class ComputeLoss(nn.Cell):
    def __init__(self, network,loss_fn):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn  
        self.reshape = P.Reshape()
    def construct(self, grp_images, grp_labels):
        cosod_maps= self.network(grp_images)
        Bg, M, _, H, W = grp_labels.shape
        cm = self.reshape(cosod_maps,(Bg*M, 1, H, W))
        gt = self.reshape(grp_labels,(Bg*M, 1, H, W))
        return  self._loss_fn(cm, gt)
    
        # return cosod_loss

train_set = 'CoSOD3k'
mean, std = dataset_mean_std(train_set)
Bg = 2
M = 5

info_file = 'Info__' + train_set + '.txt'
list_file = 'List__' + train_set + '.txt'
GroupNumbers, GroupNames, GroupSizes, GroupFileLists = CoSOD_Dataset_Parser(dataset_root, info_file, list_file)

for g in range(GroupNumbers):
    group_file_list = GroupFileLists[g]
    gs = len(group_file_list)
    if gs < M:
        for add_index in range(M-gs):
            GroupFileLists[g].append(GroupFileLists[g][add_index])
        GroupSizes[g] += (M-gs)

#%%

net = GLNet(return_loss=True)
param_dict = ms.load_checkpoint(os.path.join(checkpoints_folder, 'warehouse', 'model_init.ckpt'))
ms.load_param_into_net(net, param_dict)
iterations = 50000
show_every = 500
show_every = 100
init_lr = 5e-6
min_lr = 5e-7

lr = np.array(nn.cosine_decay_lr(0., init_lr , GroupNumbers * iterations, GroupNumbers,
                                         iterations))                   
optimizer = nn.optim.Adam(net.trainable_params(), lr, weight_decay=1e-5)
bce_loss = nn.BCEWithLogitsLoss() 
model = ComputeLoss(net,bce_loss)
T_net = nn.TrainOneStepCell(model, optimizer)
net.set_train()
#%%

record_cosod_loss = 0

for it in range(1, iterations+1):
    time_start = time.time()
    time_start = time.time()
    grp_images, grp_labels, grp_names = Random_Batch_Group_Loader(dataset_root, GroupNumbers, GroupFileLists, Bg, M, mean, std)
    grp_images, grp_labels = grp_images, grp_labels
    # print("grp_images init",grp_images.shape)
    # print("grp_labels init",grp_labels.shape)
    cosod_loss = T_net(grp_images.squeeze(), grp_labels)
    cosod_loss = cosod_loss.mean()
    record_cosod_loss += cosod_loss * (Bg*M)
    time_end = time.time()
    if it <= 20:
        print('running time per iteration: {} seconds'.format(np.around((time_end-time_start), 2)))
    # if it == 21:
    #     clear_output()
    # print(record_cosod_loss)
    if np.mod(it, show_every) == 0:
        cache_model(net, os.path.join(checkpoints_folder, 'trained', 'GLNet.ckpt'), False)
        
        record_cosod_loss = np.around(record_cosod_loss.asnumpy() / (show_every*Bg*M), 6)
        print('iteration: {}, cosod loss: {}'.format(align_number(it, 6), record_cosod_loss))
        record_cosod_loss = 0
    if it == 40000:
        cache_model(net, os.path.join(checkpoints_folder, 'trained', 'GLNet40000.ckpt'), multi_gpu)

