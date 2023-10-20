import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(512, latent_dim)

    def forward(self, x):
        e = F.relu(self.fc1(x))
        mu = self.fc2(e)
        lv = self.fc3(e)
        # return mu,lv.mul(0.5).exp_()
        return mu, torch.tensor(0.75)


class QNet2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(QNet2, self).__init__()
        self.fc11 = nn.Linear(input_dim, 512)
        self.fc22 = nn.Linear(512, latent_dim)
        self.fc33 = nn.Linear(512, latent_dim)

    def forward(self, x):
        e = F.relu(self.fc11(x))
        mu = self.fc22(e)
        lv = self.fc33(e)
        # return mu,lv.mul(0.5).exp_()
        return mu, torch.tensor(0.75)


class QNet3(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(QNet3, self).__init__()
        self.fc111 = nn.Linear(input_dim, 512)
        self.fc222 = nn.Linear(512, latent_dim)
        self.fc333 = nn.Linear(512, latent_dim)

    def forward(self, x):
        e = F.relu(self.fc111(x))
        mu = self.fc222(e)
        lv = self.fc333(e)
        # return mu,lv.mul(0.5).exp_()
        return mu, torch.tensor(0.75)
