import os
import time
import argparse
import numpy as np
from PIL import Image
from numpy import random
import cv2
# from IPython.display import clear_output

import mindspore as ms
from mindspore import nn, dataset, context, ops,ms_function,set_context,PYNATIVE_MODE
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

# import warnings
# warnings.filterwarnings('ignore')