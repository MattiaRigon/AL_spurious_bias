import torch.nn as nn
from omegaconf import MISSING
from torchvision.models import ResNet50_Weights, resnet50

from . import ModelBase


class ResNet50(ModelBase):
    pretrained: ResNet50_Weights | None = MISSING
    freeze_encoder: bool = MISSING

    def setup(self, n_classes: int):
        self._model = resnet50(weights=self.pretrained)
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
