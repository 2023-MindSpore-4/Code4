from copy import deepcopy
from mindspore import nn, train,ops
import mindspore


class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.set_train(mode=False)
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'Cell')
        self.param_keys = [k for k, _ in self.ema.parameters_and_names()]
        # self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.get_parameters():
            p.requires_grad=False

    def update(self, model):
        needs_module = hasattr(model, 'Cell') and not self.ema_has_module
        with mindspore.ops.stop_gradient():
            msd = model.parameters_dict()
            esd = self.ema.parameters_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'Cell.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            # for k in self.buffer_keys:
            #     if needs_module:
            #         j = 'Cell.' + k
            #     else:
            #         j = k
            #     esd[k].copy_(msd[j])
