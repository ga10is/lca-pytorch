from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
try:
    import matplotlib.pylab as plt
    # %matplotlib inline
except ImportError:
    print('no matplotlib')

from .db.mstory import ModelDB
from .model.metrics import AverageMeter, accuracy
from .lca import Lca


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


class LcaDataset(Dataset):
    def __init__(self, data_size):
        self.data_size = data_size
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=get_transforms('train'))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.dataset[idx]


DEVICE = torch.device("cuda:0")


def train(lca_freq):
    """
    Training

    Parameters
    ----------
    lca_freq: int
        Frequency of calculating LCA
    """
    epochs = 20

    # load dataset
    # TODO: change valid -> train
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=get_transforms('train'))
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, drop_last=True)

    # init model
    model = ResNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    lca = Lca(model, criterion)
    lca_dataset = LcaDataset(1000)
    lca_loader = DataLoader(lca_dataset, batch_size=128,
                            shuffle=True, num_workers=8, drop_last=False)
    db = ModelDB('./sample.db')
    db.rec_model(model, './some_path/')

    # lr
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    # train
    model.train()
    for epoch in range(1, epochs + 1):
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        # train one epoch
        for i, data in enumerate(tqdm(train_loader)):
            # train one iteration
            img, label = data
            img, label = img.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            logit = model(img)
            loss = criterion(logit, label)
            loss.backward()

            # record history
            acc = accuracy(logit.detach(), label.detach())[0]
            db.rec_history(epoch, i, 'train', img.size(0),
                           lr, loss.item(), acc.item())
            loss_meter.update(loss.item(), img.size(0))
            acc_meter.update(acc.item(), img.size(0))

            if (i + 1) % lca_freq == 0:
                print('loss: %.6f acc: %.6f' % (loss_meter.avg, acc_meter.avg))

                # set previous parameter(theta(t))
                lca.set_cur_theta(model)
                optimizer.step()
                # set current parameter(theta(t+1))
                lca.set_next_theta(model)

                lca_dict = lca.calc_grad(lca_loader)
                db.rec_lca(lca_dict, epoch, i)
                # plot_lca(lca_dict)
            else:
                optimizer.step()

        # valid one epoch

        # scheduler.step()


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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
