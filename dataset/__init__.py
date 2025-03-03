import logging
from abc import ABC, abstractclassmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torchvision.transforms as transforms
from hydra.utils import to_absolute_path
from typing_extensions import dataclass_transform
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset

logger = logging.getLogger(__name__)


@dataclass
@dataclass_transform()
class DatasetBase(ABC):
    root: Path
    seed: int
    name: str

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def __post_init__(self):
        self.root = Path(to_absolute_path(self.root))
        dataset: WILDSDataset = self.create()

        self.train: WILDSSubset = dataset.get_subset("train", transform=self.get_transform("train"))
        self.val: WILDSSubset = dataset.get_subset("val", transform=self.get_transform("val"))
        self.test: WILDSSubset = dataset.get_subset("test", transform=self.get_transform("test"))
        self.grouper: CombinatorialGrouper = dataset._eval_grouper

        attr_fields = deepcopy(self.grouper.groupby_fields)
        attr_fields.remove("y")
        self.attr_grouper = CombinatorialGrouper(dataset, groupby_fields=attr_fields)
        self.n_classes: int = dataset.n_classes
        logger.info(f"{self.__class__.__name__}")

    @abstractclassmethod
    def create(self):
        ...

    @property
    def get_transform(self, split: str) -> transforms.Compose | None:
        return None


from .bar import BAR  # noqa: F401, E402
from .binary_cmnist import BinaryCMNIST  # noqa: F401, E402
from .celeba import CelebA  # noqa: F401, E402
from .cifar10 import CIFAR10  # noqa: F401, E402
from .cmnist import ColoredMNIST  # noqa: F401, E402
from .corrupted_cifar10 import CorruptedCIFAR10  # noqa: F401, E402
from .imb_binary_cmnist import ImBBinaryCMNIST  # noqa: F401, E402
from .metashift import MetaShift  # noqa: F401, E402
from .svhn import SVHN  # noqa: F401, E402
from .treeperson import TreePerson  # noqa: F401, E402
from .waterbirds import WaterBirds  # noqa: F401, E402
