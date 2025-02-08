import logging
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
import torch.multiprocessing
from environs import Env
from typing_extensions import dataclass_transform

from dataset import DatasetBase
from model import ModelBase

logger = logging.getLogger(__name__)


class WandBModeEnum(Enum):
    online = auto()
    offline = auto()
    disabled = auto()


@dataclass
class WandBConfig:
    mode: WandBModeEnum
    project: str
    entity: str
    tags: Optional[list[str]]

    def __post_init__(self):
        self.mode = self.mode.name


@dataclass
@dataclass_transform()
class JobSimple(ABC):
    seed: int
    wandb: WandBConfig

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def __post_init__(self):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
        except AssertionError:
            pass

    @abstractclassmethod
    def run(self): ...


@dataclass
@dataclass_transform()
class JobBase(ABC):
    seed: int
    dataset: DatasetBase
    model: ModelBase
    wandb: WandBConfig

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def __post_init__(self):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
        except AssertionError:
            pass

        self.model.setup(self.dataset.n_classes)
        if Env().bool("USE_CUDA") and torch.cuda.is_available():
            self.model.cuda()
            logger.info("using cuda....")
        elif torch.backends.mps.is_available():
            self.model.to(torch.device("mps"))
            logger.info("using MPS....")
        else:
            logger.warning("using cpu....")

    @abstractclassmethod
    def run(self): ...


from .active import ActiveLearning  # noqa: F401, E402
from .dynamics import LearningDynamics  # noqa: F401, E402
from .loss_landscape import LossLandscape  # noqa: F401, E402
from .passive import ERM  # noqa: F401, E402
from .trajectory import Trajectory  # noqa: F401, E402
