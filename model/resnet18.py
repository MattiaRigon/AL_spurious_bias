from typing import Optional
import torch.nn as nn
from omegaconf import MISSING
from torchvision.models import ResNet18_Weights, resnet18

from . import ModelBase


class ResNet18(ModelBase):
    pretrained: Optional[ResNet18_Weights] = MISSING
    freeze_encoder: bool = MISSING

    def setup(self, n_classes: int):
        self._model = resnet18(weights=self.pretrained)
        self._encoder = self._model
        self._clf = nn.Linear(self._model.fc.in_features, n_classes)
        self._encoder.fc = nn.Identity()
        if self.freeze_encoder:
            for _, param in self._encoder.named_parameters():
                param.requires_grad = False

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._clf(self._encoder(x))
        embed = self._encoder(x)
        return embed, self._clf(embed)
