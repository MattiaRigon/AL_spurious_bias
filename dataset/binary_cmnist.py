import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset
from typing import Optional
from . import DatasetBase


class BinaryCMNISTBase(WILDSDataset):
    _dataset_name = "binary-cmnist"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir, download, seed, confounding_factor) -> None:
        self._data_dir = root_dir
        train = MNIST(root=root_dir, train=True, download=download)
        test = MNIST(root=root_dir, train=False, download=download)

        assert 0 <= confounding_factor <= 1

        train.targets = (train.targets >= 5).to(torch.int64)
        test.targets = (test.targets >= 5).to(torch.int64)

        train.data = torch.einsum("ijk,c -> ijkc", train.data, torch.tensor([1, 1, 1], dtype=torch.uint8))
        test.data = torch.einsum("ijk,c -> ijkc", test.data, torch.tensor([1, 1, 1], dtype=torch.uint8))

        rng = np.random.RandomState(42)

        def generate_attr(targets, alpha):
            attr = torch.zeros_like(targets)
            for y, a in zip([0, 1], [alpha, 1 - alpha]):
                idx = torch.where(targets == y)[0]
                s = rng.binomial(1, a, len(idx))
                attr[idx] = torch.from_numpy(s)
            return attr

        train.attr = generate_attr(train.targets, confounding_factor)
        test.attr = generate_attr(test.targets, 0.5)

        mask = torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.uint8)
        train.data = torch.einsum("ijkc,ic -> ijkc", train.data, mask[train.attr])
        test.data = torch.einsum("ijkc,ic -> ijkc", test.data, mask[test.attr])

        self._input_array = np.vstack([train.data, test.data, test.data])
        self._y_array = torch.concat([train.targets, test.targets, test.targets])
        self._y_size = 1
        self._n_classes = 2

        attr = torch.concat([train.attr, test.attr, test.attr])
        self._metadata_array = torch.stack((attr, self._y_array), dim=1)
        self._metadata_fields = ["colour", "y"]
        self._metadata_map = {
            "colour": [" red", "blue"],  # Padding for str formatting
            "y": [" < 5", ">=5"],
        }

        self._split_scheme = "official"
        self._split_array = np.zeros_like(self._y_array)
        # dummy val set
        self._split_array = np.array(
            [0] * len(train.targets) + [1] * len(test.targets) + [2] * len(test.targets),
            dtype=np.int64,
        )

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["colour", "y"]))

        super().__init__(root_dir, download, None)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img = self._input_array[idx]
        img = Image.fromarray(img)
        return img


class BinaryCMNIST(DatasetBase):
    skewed_ratio: float
    subsample_train: Optional[int]

    def create(self):
        dataset = BinaryCMNISTBase(
            download=True,
            root_dir=self.root,
            seed=self.seed,
            confounding_factor=self.skewed_ratio,
        )
        if self.subsample_train is not None:
            train_idx = np.where(dataset._split_array == dataset.DEFAULT_SPLITS["train"])[0]
            chosen = np.random.RandomState(42).choice(train_idx, self.subsample_train, replace=False)
            to_remove = np.setdiff1d(train_idx, chosen)
            dataset._split_array[to_remove] = -1
        return dataset

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
            ]
        )
