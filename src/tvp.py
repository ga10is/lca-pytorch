from tqdm import tqdm
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
try:
    import matplotlib.pylab as plt
    # %matplotlib inline
except ImportError:
    print('no matplotlib')

from .db.mstory import ModelDB


def get_transforms(mode):

    if mode == 'train':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])
    return transforms


def train(lca_freq):
    """
    Training

    Parameters
    ----------
    lca_freq: int
        Frequency of calculating LCA
    """

    epochs = 1

    # load dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=get_transforms('train'))
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=4)

    # init model
    model = MyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    lca = Lca(model, criterion)
    db = ModelDB('./sample.db')
    db.rec_model(model, './some_path/')

    # train
    model.train()
    for epoch in range(1, epochs + 1):
        # train one epoch
        for i, data in enumerate(tqdm(train_loader)):
            # train one iteration
            img, label = data

            optimizer.zero_grad()

            logit = model(img)
            loss = criterion(logit, label)
            loss.backward()

            if i % lca_freq == 0:
                # set previous parameter(theta(t))
                lca.set_cur_theta(model)
                optimizer.step()
                # set current parameter(theta(t+1))
                lca.set_next_theta(model)

                lca_dict = lca.calc_grad(img, label)
                db.rec_lca(lca_dict, epoch, i)
                # plot_lca(lca_dict)
            else:
                optimizer.step()

        # valid one epoch

        # scheduler.step()


class Lca:
    # TODO: introduce with
    # with lca.set_theta():
    #     optimizer.step()

    def __init__(self, model, criterion):
        """
        Parameters
        ----------
        model_org: torch.nn.Module
            model to set parameters, and calculate loss/gradient.
        criterion: torch.nn.Module
            loss function
        """
        self.model = copy.deepcopy(model)
        self.criterion = copy.deepcopy(criterion)
        self.theta_list = [None, None, None]

    def set_cur_theta(self, model):
        # name_param_dict = OrderedDict()
        # for n, p in model.named_parameters():
        # name_param_dict[n] = p.data
        self.theta_list[0] = copy.deepcopy(model.state_dict(keep_vars=False))

    def set_next_theta(self, model):
        self.theta_list[-1] = copy.deepcopy(model.state_dict(keep_vars=False))

    def set_fractional_theta(self):
        if self.theta_list[0] is None:
            raise ValueError('Current parameters must be set')
        if self.theta_list[-1] is None:
            raise ValueError('Current parameters must be set')

        theta_frac = OrderedDict()
        for (n1, p1), (n2, p2) in zip(self.theta_list[0].items(), self.theta_list[-1].items()):
            if n1 != n2:
                raise ValueError(
                    'Names of current and next parameter are different')
            # 1/2 theta_t + 1/2 theta_(t+1)
            # theta_frac[n1] = torch.add(p1.data, p2.data) / 2
            theta_frac[n1] = (p1.data + p2.data) / 2

        self.theta_list[1] = theta_frac

    def calc_grad(self, x, y):
        """
        Calculate LCA for each layer.

        Parameters
        ----------
        x: torch.Tensor
            input of model
        y: torch.Tensor
            output of model

        Returns
        -------
        OrderedDict
            key: parameter name. e.g. conv1.weight
            vlaue: LCA for each parameter
        """

        lca = OrderedDict()
        coeffs = [1, 4, 1]

        # set 1/2 theta_t + 1/2 theta_(t+1)
        self.set_fractional_theta()

        loss_vals = []
        for theta, coeff in zip(self.theta_list, coeffs):
            # set parameter to model
            self.model.load_state_dict(theta)

            self.model.zero_grad()
            logit = self.model(x)
            loss = self.criterion(logit, y)
            loss_vals.append(loss.item())
            loss.backward()

            # calculate LCA
            for n, p in self.model.named_parameters():
                if n not in lca:
                    lca[n] = coeff * p.grad.data / sum(coeffs)
                else:
                    lca[n] += coeff * p.grad.data / sum(coeffs)

        print(loss_vals[-1] - loss_vals[0])

        for k, v in lca.items():
            # print(k, v)
            lca[k] *= (self.theta_list[-1][k] - self.theta_list[0][k])

        return lca


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def plot_lca(lca_dict, figsize=(16, 6)):
    sum_lca = [lca_dict[k].sum().item() for k in lca_dict.keys()]
    layers = list(lca_dict.keys())

    print(sum_lca)

    plt.figure(figsize=figsize)
    plt.bar(x=np.arange(len(layers)), height=sum_lca,
            alpha=0.5, lw=1, color='c')
    plt.xticks(range(0, len(layers), 1), layers, rotation='vertical')
    plt.xlim(left=-1, right=len(layers))
    # plt.ylim(-1, 0)
    plt.xlabel("Layers")
    plt.ylabel("Loss Change Allocation")
    plt.title("Gradient flow")
    plt.grid(True)
