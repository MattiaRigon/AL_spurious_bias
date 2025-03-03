import torchvision.transforms as transforms
import numpy as np
from .celebA_dataset import CelebADataset
from . import DatasetBase
from enum import Enum, auto


class TargetEnum(Enum):
    Blond_Hair = auto()
    Black_Hair = auto()
    Brown_Hair = auto()
    Smiling = auto()


class CelebA(DatasetBase):
    target: TargetEnum
    subsample_train: int | None

    def create(self):
        dataset = CelebADataset(root_dir=self.root, download=True, target_name=self.target.name)
        if self.subsample_train is not None:
            train_idx = np.where(dataset._split_array == dataset.DEFAULT_SPLITS["train"])[0]
            chosen = np.random.RandomState(42).choice(train_idx, self.subsample_train, replace=False)
            to_remove = np.setdiff1d(train_idx, chosen)
            dataset._split_array[to_remove] = -1
        return dataset

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
