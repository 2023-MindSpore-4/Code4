import os
import json
import shutil
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint as ms_save_checkpoint


def save_checkpoint(state, is_best, args, filename='default'):
    if filename == 'default':
        filename = 'STAN_%s_batch%d' % (args.dataset, args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.ckpt' % (filename)
    # save_list = [{'name': k, 'data': v} for k, v in state.items() if k in ['state_dict']]
    my_dict = {k: state[k] for k in ['epoch', 'best_loss']}
    # my_dict.update({'optimizer': json.dumps(state['optimizer'])})
    best_name = './saved_models/%s_model_best.ckpt' % (filename)
    ms_save_checkpoint([], checkpoint_name, append_dict=my_dict)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)


def load_pretrain(model, args, logging):
    if os.path.isfile(args.pretrain):
        param_dict = load_checkpoint(args.pretrain)
        load_param_into_net(model, param_dict)
        print("=> loaded pretrain model at {}".format(args.pretrain))
        logging.info("=> loaded pretrain model at {}".format(args.pretrain))
    else:
        print("=> no pretrained file found at '{}'".format(args.pretrain))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    return model


def load_resume(model, args, logging):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        param_dict = load_checkpoint(args.resume)
        args.start_epoch = param_dict['epoch']
        best_loss = param_dict['best_loss']
        load_param_into_net(model, param_dict)
        print("=> loaded checkpoint (epoch {}) Loss{}".format(param_dict['epoch'], best_loss))
        logging.info("=> loaded checkpoint (epoch {}) Loss{}".format(param_dict['epoch'], best_loss))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model


def load_resume_optimizer(optimizer, args, logging):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        param_dict = load_checkpoint(args.resume)
        optimizer.load_state_dict(param_dict['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return optimizer
