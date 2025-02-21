from typing import Optional
import torch
import torch.nn as nn
from omegaconf import MISSING
from torchvision.models import ResNet50_Weights, resnet50
from torch.optim import SGD, Adam
from . import ModelBase
from .optim import OptimizerConfig  # Import OptimizerConfig

class ResNet50(ModelBase):
    pretrained: Optional[ResNet50_Weights] = MISSING
    freeze_encoder: bool = MISSING
    optim: OptimizerConfig = MISSING  # Use OptimizerConfig instead of dict

    def setup(self, n_classes: int):
        self._model = resnet50(weights=self.pretrained)
        self._encoder = self._model
        self._clf = nn.Linear(self._model.fc.in_features, n_classes)
        self._encoder.fc = nn.Identity()

        # Activation storage
        self.activations = None

        # Hook to store activations from the last convolutional layer
        def save_activation(module, input, output):
            self.activations = output  # Save feature maps
        
        # Register hook on last convolutional layer
        self._encoder.layer4[-1].register_forward_hook(save_activation)

        # Freeze encoder if specified
        if self.freeze_encoder:
            for _, param in self._encoder.named_parameters():
                param.requires_grad = False

        # Select and instantiate optimizer based on config
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

    def __call__(self, x, return_embedding=False, output_activations=False):
        self.activations = None  # Reset activations at each forward pass
        
        if not return_embedding:
            logits = self._clf(self._encoder(x))
            return (logits, self.activations) if output_activations else logits

        embed = self._encoder(x)
        return (embed, self._clf(embed), self.activations) if output_activations else (embed, self._clf(embed))
