import argparse

__all__ = ['set_parser']

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch OpenMatch Training')
    ## Computational Configurations
    parser.add_argument('--gpu_id', default='0', type=int,
                        help=''
                             '')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O0",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training: local_rank")
    parser.add_argument('--no_progress', action='store_true',
                    help="don't use progress bar")
    parser.add_argument('--eval_only', type=int, default=0,
                        help='1 if evaluation mode ')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='for cifar10')

    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./data/cifar-10-binary/cifar-10-batches-bin', type=str,
                        help='path to data directory')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='dataset name')
    ## Hyper-parameters
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='optimize name')
    parser.add_argument('--num-labeled', type=int, default=400,
                        choices=[25, 50, 100, 400],
                        help='number of labeled data per each class')
    parser.add_argument('--num_val', type=int, default=50,
                        help='number of validation data per each class')
    parser.add_argument('--num_super', type=int, default=10,
                        help='number of super-class known classes cifar100: 10 or 15')
    parser.add_argument("--expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext',
                                 'resnet_imagenet'],
                        help='dataset name')
    ## HP unique to OpenMatch (Some are changed from FixMatch)
    parser.add_argument('--lambda_oem', default=0.1, type=float,
                    help='coefficient of OEM loss')
    parser.add_argument('--lambda_socr', default=0.5, type=float,
                    help='coefficient of SOCR loss, 0.5 for CIFAR10, ImageNet, '
                         '1.0 for CIFAR100')
    parser.add_argument('--start_fix', default=10, type=int,
                        help='epoch to start fixmatch training')
    parser.add_argument('--mu', default=2, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--total_steps', default=2 ** 19, type=int,
                        help='number of total steps to run')
    parser.add_argument('--epochs', default=512, type=int,
                        help='number of epochs to run')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='pseudo label threshold')
    ##
    parser.add_argument('--eval_step', default=1024, type=int,
                        help='number of eval steps to run')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=2, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--aug_num', default=3, type=int,
                        help='number of augmentation')
    parser.add_argument('--disentangle', default=False, action='store_true')
    parser.add_argument('--rec_coef', default=0.1, type=float,
                        help="coefficient for reconstruction loss")
    parser.add_argument('--lambda_disent', default=0.08, type=float,
                    help='coefficient of disentaglement loss')
    parser.add_argument('--lambda_vae', default=0.001, type=float,
                    help='coefficient of vae loss')
    parser.add_argument('--start_augmentation', default=128, type=int,
                    help='epochs for learning with augmented data')
    parser.add_argument('--augmentation_round', default=4, type=int,
                    help='how many epoch rounds for learning with augmented data')
    parser.add_argument('--augment', default=False, action='store_true')
    # parser.add_argument('--start_augmentation', default=256, type=int,
    #                 help='epoch interval for different augmentation round')
    parser.add_argument('--augmentation_interval', default=16, type=int,
                    help='epoch interval for different augmentation round')
    parser.add_argument('--adv_eps', default=0.03, type=float,
                    help='maximum value of adversarial perturbation')
    parser.add_argument('--adv_magnitude', default=3e-4, type=float,
                    help='magnitude of adversarial perturbation')
    parser.add_argument('--adv_step', default=10, type=int,
                    help='step of adversarial perturbation')
    parser.add_argument('--lambda_aug', default=0.4, type=float,
                    help='lambda for adversarial loss')


    args = parser.parse_args()
    return args