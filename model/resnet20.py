import torch.nn as nn
from omegaconf import MISSING

from . import ModelBase
from .resnet import resnet20


class ResNet20(ModelBase):
    freeze_encoder: bool = MISSING

    def setup(self, n_classes: int):
        self._model = resnet20(n_classes)
        self._encoder = self._model
        self._clf = nn.Linear(self._model.linear.in_features, n_classes)
        self._encoder.linear = nn.Identity()
        if self.freeze_encoder:
            for _, param in self._encoder.named_parameters():
                param.requires_grad = False

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._clf(self._encoder(x))
        embed = self._encoder(x)
        return embed, self._clf(embed)
