import torch.nn as nn

from . import ModelBase


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        out = self.classifier(feat)
        return out


class MLP(ModelBase):
    def setup(self, n_classes: int):
        self._model = Net(n_classes)

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._model(x)
        x = x.view(x.size(0), -1) / 255
        embed = self._model.feature(x)
        return embed, self._model.classifier(embed)
