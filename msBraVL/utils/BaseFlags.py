import os
import argparse
import msadapter.pytorch as torch
import scipy.io as sio
parser = argparse.ArgumentParser()

# TRAINING
parser.add_argument('--batch_size', type=int, default=512, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=32, help="dimension of common factor latent space")
# SAVE and LOADBraVL/BraVL_fMRI/data
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# DIRECTORIES
# experiments
parser.add_argument('--dir_experiment', type=str, default='./logs', help="directory to save logs in")
parser.add_argument('--dataname', type=str, default='DIR-Wiki', help="dataset")
parser.add_argument('--sbj', type=str, default='sub-03', help="fmri subject")
parser.add_argument('--roi', type=str, default='LVC_HVC_IT', help="ROI")
parser.add_argument('--text_model', type=str, default='GPTNeo', help="text embedding model")
parser.add_argument('--image_model', type=str, default='pytorch/repvgg_b3g4', help="image embedding model")
parser.add_argument('--stability_ratio', type=str, default='', help="stability_ratio")
parser.add_argument('--test_type', type=str, default='zsl', help='normal or zsl')
parser.add_argument('--aug_type', type=str, default='image_text', help='no_aug, image_text, image_only, text_only')
#multimodal
parser.add_argument('--method', type=str, default='joint_elbo', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=0.0, help="default initial weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")
parser.add_argument('--lambda1', type=float, default=0.001, help="default weight of intra_mi terms")
parser.add_argument('--lambda2', type=float, default=0.001, help="default weight of inter_mi terms")


FLAGS = parser.parse_args()
data_dir_root = os.path.join('/home/chem/Desktop/BraVL/BraVL_fMRI/data', FLAGS.dataname)
brain_dir = os.path.join(data_dir_root, 'brain_feature', FLAGS.roi, FLAGS.sbj)
image_dir_train = os.path.join(data_dir_root, 'visual_feature/ImageNetTraining', FLAGS.image_model+'-PCA', FLAGS.sbj)
text_dir_train = os.path.join(data_dir_root, 'textual_feature/ImageNetTraining/text', FLAGS.text_model, FLAGS.sbj)

train_brain = sio.loadmat(os.path.join(brain_dir, 'fmri_train_data'+FLAGS.stability_ratio+'.mat'))['data'].astype('double')
train_image = sio.loadmat(os.path.join(image_dir_train, 'feat_pca_train.mat'))['data'].astype('double')#[:,0:3000]
train_text = sio.loadmat(os.path.join(text_dir_train, 'text_feat_train.mat'))['data'].astype('double')
train_brain = torch.from_numpy(train_brain)
train_image = torch.from_numpy(train_image)
train_text = torch.from_numpy(train_text)
dim_brain = train_brain.shape[1]
dim_image = train_image.shape[1]
dim_text = train_text.shape[1]

parser.add_argument('--m1_dim', type=int, default=dim_brain, help="dimension of modality brain")
parser.add_argument('--m2_dim', type=int, default=dim_image, help="dimension of modality image")
parser.add_argument('--m3_dim', type=int, default=dim_text, help="dimension of modality text")
parser.add_argument('--data_dir_root', type=str, default=data_dir_root, help="data dir")

FLAGS = parser.parse_args()
# print(FLAGS)
