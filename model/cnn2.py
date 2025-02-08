import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ModelBase


class Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)

    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def clf(self, x):
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        return self.clf(x)


class CNN2(ModelBase):
    def setup(self, n_classes: int):
        self._model = Net(n_classes)

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._model(x)
        embed = self._model.encoder(x)
        return embed, self._model.clf(embed)
