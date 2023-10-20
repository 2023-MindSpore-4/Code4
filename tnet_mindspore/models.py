import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_de3 = nn.Conv2dTranspose(in_channels=128, out_channels=64, kernel_size=4, stride=2)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv_de4 = nn.Conv2dTranspose(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv_6 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv_22 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        
        self.act = nn.LeakyReLU()
        self.cat=ops.Concat(axis=1)

    def construct(self, x):
        #1
        c_1 = self.act(self.conv_1(x))
        c_p_1 = self.pool_1(c_1)
        c_2 = self.act(self.conv_2(c_p_1))
        c_p_2 = self.pool_2(c_2)
        c_3 = self.act(self.conv_3(c_p_2))
        up_3 = self.conv_de3(c_3)
        cat_23 = self.cat([up_3, c_2])
        c_4 = self.act(self.conv_4(cat_23))
        up_4 = self.conv_de4(c_4)
        cat_14 = self.cat([up_4, c_1])
        c_5 = self.act(self.conv_5(cat_14))
        c_6 = self.conv_6(c_5)
        R_out = self.sigmoid(c_6)
        
        c_21 = self.act(self.conv_21(c_1))
        cat_215 = self.cat([c_21, c_5])
        c_22 = self.conv_22(cat_215)
        L_out = self.sigmoid2(c_22)
        L_out_3 = self.cat([L_out, L_out, L_out])

        return R_out, L_out_3

if __name__ == '__main__':
    x = np.ones([2, 3, 256, 256])
    x = mindspore.Tensor(x, mindspore.float32)
    net = Net()
    out = net(x)
    print(out)