import mindspore as ms
import numpy as np
import mindspore.nn.probability.distribution as dist
import msadapter.pytorch as torch

a = dist.Normal(0,1)
print(a.mean().cpu().numpy())

