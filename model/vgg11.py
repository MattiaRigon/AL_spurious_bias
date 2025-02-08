# from https://github.com/haonan3/Deep-Active-Learning-by-Leveraging-Training-Dynamics/blob/main/src/models.py

import torch.nn as nn

from . import ModelBase


class _VGG11(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        in_channels = 3
        x = 64
        self.cn1 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(x)
        self.relu1 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        x = 128
        self.cn2 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(x)
        self.relu2 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        x = 256
        self.cn3 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(x)
        self.relu3 = nn.ReLU(inplace=True)
        in_channels = x
        x = 256
        self.cn4 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(x)
        self.relu4 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        x = 512
        self.cn5 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(x)
        self.relu5 = nn.ReLU(inplace=True)
        in_channels = x
        x = 512
        self.cn6 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(x)
        self.relu6 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        x = 512
        self.cn7 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(x)
        self.relu7 = nn.ReLU(inplace=True)
        in_channels = x
        x = 512
        self.cn8 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(x)
        self.relu8 = nn.ReLU(inplace=True)
        self.mpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat_extractor = nn.Sequential(
            self.cn1,
            self.bn1,
            self.relu1,
            self.mpool1,
            self.cn2,
            self.bn2,
            self.relu2,
            self.mpool2,
            self.cn3,
            self.bn3,
            self.relu3,
            self.cn4,
            self.bn4,
            self.relu4,
            self.mpool3,
            self.cn5,
            self.bn5,
            self.relu5,
            self.cn6,
            self.bn6,
            self.relu6,
            self.mpool4,
            self.cn7,
            self.bn7,
            self.relu7,
            self.cn8,
            self.bn8,
            self.relu8,
            self.mpool4,
        )

        self.linear = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x

    def predict(self, out):
        out = self.linear(out)
        return out

    def get_embedding(self, x):
        out = self.feat_extractor(x)
        # out = self.features(x)
        emb = out.view(out.size(0), -1)
        return emb


class VGG11(ModelBase):
    def setup(self, n_classes: int):
        self._model = _VGG11(n_classes)

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._model(x)
        embed = self._model.get_embedding(x)
        return embed, self._model.predict(embed)
