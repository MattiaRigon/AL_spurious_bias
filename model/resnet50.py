from typing import Optional
import torch.nn as nn
from omegaconf import MISSING
from torchvision.models import ResNet50_Weights, resnet50
from torch.optim import SGD, Adam  # Importa gli ottimizzatori che userai
from . import ModelBase
from .optim import OptimizerConfig  # Import OptimizerConfig

from torch.optim import SGD, Adam
from . import ModelBase

class ResNet50(ModelBase):
    pretrained: Optional[ResNet50_Weights] = MISSING
    freeze_encoder: bool = MISSING
    optim: OptimizerConfig = MISSING  # Use OptimizerConfig instead of dict

    def setup(self, n_classes: int):
        self._model = resnet50(weights=self.pretrained)
        self._encoder = self._model
        self._clf = nn.Linear(self._model.fc.in_features, n_classes)
        self._encoder.fc = nn.Identity()
        if self.freeze_encoder:
            for _, param in self._encoder.named_parameters():
                param.requires_grad = False

        # Seleziona e istanzia l'ottimizzatore in base alla configurazione
        optim_config = self.optim
        if optim_config.get_target() == "model.optim.SGD":
            self.optimizer = SGD(
                self._model.parameters(),
                lr=optim_config["lr"],
                momentum=optim_config["momentum"],
                weight_decay=optim_config["weight_decay"]
            )
        elif optim_config.get_target() == "model.optim.Adam":
            self.optimizer = Adam(self._model.parameters(), lr=optim_config["lr"])
        else:
            raise ValueError(f"Optimizer {optim_config['_target_']} not supported")

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self._clf(self._encoder(x))
        embed = self._encoder(x)
        return embed, self._clf(embed)

