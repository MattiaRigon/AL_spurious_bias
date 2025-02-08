import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10 as _CIFAR10
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase


class CIFAR10Base(WILDSDataset):
    _dataset_name = "cifar10"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir, download, seed) -> None:
        self._data_dir = root_dir
        train = _CIFAR10(root=root_dir, train=True, download=download)
        test = _CIFAR10(root=root_dir, train=False, download=download)

        # self._y_array = torch.LongTensor(train.targets + test.targets)
        # dummy val set
        self._y_array = torch.LongTensor(train.targets + test.targets + test.targets)
        self._y_size = 1
        self._n_classes = len(torch.unique(self._y_array))

        self._metadata_array = torch.stack((torch.zeros_like(self._y_array), self._y_array), dim=1)
        self._metadata_fields = ["dummy", "y"]
        self._metadata_map = {
            "dummy": [" "],  # Padding for str formatting
            "y": [" airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"],
        }

        self._input_array = np.vstack([train.data, test.data, test.data])
        self._original_resolution = (32, 32)

        self._split_scheme = "official"
        # rng = np.random.RandomState(seed)
        # split_ind = rng.permutation(len(self._y_array))
        self._split_array = np.zeros_like(self._y_array)
        # dummy val set
        self._split_array = np.array(
            [0] * len(train.targets) + [1] * len(test.targets) + [2] * len(test.targets),
            dtype=np.int64,
        )
        # r_train = round(0.6 * len(self._y_array))
        # r_val = round(0.2 * len(self._y_array))
        # self._split_array[split_ind[:r_train]] = 0
        # self._split_array[split_ind[r_train : r_val + r_train]] = 1  # noqa: E203
        # self._split_array[split_ind[r_val + r_train :]] = 2  # noqa: E203

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["dummy", "y"]))

        super().__init__(root_dir, True, "official")

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = self._input_array[idx]
        img = Image.fromarray(img)
        return img


class CIFAR10(DatasetBase):
    def create(self):
        return CIFAR10Base(download=True, root_dir=self.root, seed=self.seed)

    def get_transform(self, split):
        if split in ["test", "val"]:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        elif split == "train":
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        else:
            raise ValueError(f"{split} not found")
