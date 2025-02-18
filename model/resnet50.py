from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torchvision.models import ResNet50_Weights, resnet50
from torch.optim import SGD, Adam
from . import ModelBase
from .optim import OptimizerConfig

class ResNet50(ModelBase):
    pretrained: Optional[ResNet50_Weights] = MISSING
    freeze_encoder: bool = MISSING
    optim: OptimizerConfig = MISSING  

    def setup(self, n_classes: int):
        self._model = resnet50(weights=self.pretrained)
        self._encoder = self._model
        self._clf = nn.Linear(self._model.fc.in_features, n_classes)
        self._encoder.fc = nn.Identity()

        # Register hook for Grad-CAM
        self.gradients = None  # Store gradients for Grad-CAM
        self.feature_maps = None  # Store feature maps for Grad-CAM

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks in the last convolutional layer
        target_layer = self._encoder.layer4[-1].conv3  # Last conv layer of ResNet50
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if self.freeze_encoder:
            for _, param in self._encoder.named_parameters():
                param.requires_grad = False

        # Setup optimizer
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

    def __call__(self, x, return_embedding=False, return_gradcam=False, target_class=None):
        output = self._clf(self._encoder(x))
        
        if return_embedding:
            embed = self._encoder(x)
            return embed, output

        if return_gradcam:
            if target_class is None:
                target_class = output.argmax(dim=1)  # Get class index for each sample in batch

            gradcam_heatmaps = self.compute_gradcam(x, target_class)
            return output, gradcam_heatmaps
        
        return output


    def compute_gradcam(self, x, target_classes):
        """
        Compute Grad-CAM heatmaps for a batch of images.
        
        Args:
            x (torch.Tensor): Input batch of images.
            target_classes (torch.Tensor): Tensor of class indices for each image in the batch.
        
        Returns:
            torch.Tensor: Grad-CAM heatmaps of shape (B, 1, H, W).
        """
        self._model.zero_grad()
        output = self._clf(self._encoder(x))

        # Select the class scores for each image in the batch
        class_scores = output.gather(1, target_classes.view(-1, 1))  # Shape: [B, 1]

        # Compute gradients for each class score with respect to feature maps
        gradients = torch.autograd.grad(outputs=class_scores, 
                                        inputs=self.feature_maps,
                                        grad_outputs=torch.ones_like(class_scores),  
                                        retain_graph=True, 
                                        create_graph=True)[0]  # Shape: [B, C, H, W]

        # Compute Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # Weighted sum
        cam = F.relu(cam)  # Apply ReLU

        # Normalize heatmaps
        cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)  # Avoid division by zero

        return cam  # Shape: (B, 1, H, W)
