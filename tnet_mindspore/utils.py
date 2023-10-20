from mindspore import ops
from mindspore.ops.function import broadcast_to

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def z_repeat(a,h,w):
    b=a.shape[0]
    n=a.shape[1]
    a=ops.expand_dims(a, 2)
    a=ops.expand_dims(a, 3)
    shape = (b, n, h, w)
    output = broadcast_to(a, shape)
    return output