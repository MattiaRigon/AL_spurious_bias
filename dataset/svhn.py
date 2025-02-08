import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import SVHN as _SVHN
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase


class SVHNBase(WILDSDataset):
    _dataset_name = "svhn"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir, download, seed):
        self._data_dir = root_dir
        train = _SVHN(root=root_dir, split="train", download=download)
        test = _SVHN(root=root_dir, split="test", download=download)

        self._y_array = torch.from_numpy(np.concatenate([train.labels, test.labels]))
        self._y_size = 1
        self._n_classes = len(torch.unique(self._y_array))

        self._metadata_array = torch.stack((torch.zeros_like(self._y_array), self._y_array), dim=1)
        self._metadata_fields = ["dummy", "y"]
        self._metadata_map = {
            "dummy": [" "],  # Padding for str formatting
            "y": [" 0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        }

        self._input_array = np.vstack([train.data, test.data])
        self._original_resolution = (32, 32)

        self._split_scheme = "official"
        rng = np.random.RandomState(seed)
        split_ind = rng.permutation(len(self._y_array))
        self._split_array = np.zeros_like(split_ind)
        r_train = round(0.6 * len(self._y_array))
        r_val = round(0.2 * len(self._y_array))
        self._split_array[split_ind[:r_train]] = 0
        self._split_array[split_ind[r_train : r_val + r_train]] = 1  # noqa: E203
        self._split_array[split_ind[r_val + r_train :]] = 2  # noqa: E203

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["dummy", "y"]))

        super().__init__(root_dir, download, "official")

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = self._input_array[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        return img


class SVHN(DatasetBase):
    def create(self):
        return SVHNBase(download=True, root_dir=self.root, seed=self.seed)

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ]
        )
