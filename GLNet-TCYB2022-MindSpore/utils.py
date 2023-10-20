from packages import *


def US2(x):
    return F.interpolate(x, scales = (1.,1.,2.,2.), mode='bilinear')


def US4(x):
    return F.interpolate(x, scales = (1.,1.,4.,4.), mode='bilinear')


def US8(x):
    return F.interpolate(x, scales = (1.,1.,8.,8.), mode='bilinear')


def DS2(x):
    return F.max_pool2d(x, (2, 2))


def DS4(x):
    return F.max_pool2d(x, (4, 4))


def DS8(x):
    return F.max_pool2d(x, (8, 8))


def align_number(number, N):
    assert type(number) == int
    num_str = str(number)
    assert len(num_str) <= N
    return (N-len(num_str))*'0' + num_str




def unload(x):
    y = x.squeeze().asnumpy()
    return y


def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed


def convert2img(x):
    return Image.fromarray(x*255).convert('L')


def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if smap.max() <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)


def cache_model(model, path, multi_gpu):
    ms.save_checkpoint(model, path)
    # if multi_gpu:
    #     torch.save(model.module.state_dict(), path)
    # else:
    #     torch.save(model.state_dict(), path)
        
        
