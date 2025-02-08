import logging
from abc import ABC, abstractclassmethod
from dataclasses import asdict, dataclass, field

import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional
from typing_extensions import dataclass_transform

from .optim import OptimizerConfig, SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
@dataclass_transform()
class ModelBase(ABC, torch.nn.Module):
    name: str = MISSING
    scheduler: Optional[SchedulerConfig] = field(default=None)

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls, eq=False)

    def __str__(self):
        params = asdict(self)
        name = self.__class__.__name__
        del params["optim"]
        del params["name"]
        return f"{name}({', '.join([f'{f}={value}' for f, value in params.items()])})"

    def __post_init__(self):
        super().__init__()
        logger.info(f"{self}")
        logger.info(f"{self.optim}")

    @abstractclassmethod
    def __call__(self, x, return_embedding=False) -> torch.TensorType:
        ...

    @abstractclassmethod
    def setup(self, n_classes: int) -> None:
        ...

    def get_optimizer(self) -> tuple[Optimizer, Optional[LRScheduler]]:
        optimizer = instantiate(self.optim, params=self.parameters())
        if self.scheduler is not None:
            scheduler = instantiate(self.scheduler, optimizer)
            return optimizer, scheduler
        return optimizer, None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


from .cnn import CNN  # noqa: F401, E402
from .cnn2 import CNN2  # noqa: F401, E402
from .mlp import MLP  # noqa: F401, E402
from .resnet18 import ResNet18  # noqa: F401, E402
from .resnet20 import ResNet20  # noqa: F401, E402
from .resnet50 import ResNet50  # noqa: F401, E402
from .vgg11 import VGG11  # noqa: F401, E402
