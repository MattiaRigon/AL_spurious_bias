import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import MNIST
from tqdm import tqdm
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase
from .colored_mnist_protocol import COLORED_MNIST_PROTOCOL
from .corrupted_cifar10 import make_attr_labels


class CMNISTBase(WILDSDataset):
    _dataset_name = "cmnist"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir: Path, download, seed, skewed_ratio, severity):
        colored_mnist_dir = root_dir / f"ColoredMNIST-Skewed{skewed_ratio}-Severity{severity}"
        os.makedirs(colored_mnist_dir, exist_ok=True)

        self._data_dir = colored_mnist_dir
        f = open(Path(self._data_dir) / "RELEASE_v1.0.txt", "a")
        f.close()

        attr_names = ["digit", "color"]
        attr_names_path = colored_mnist_dir / "attr_names.pkl"
        with open(attr_names_path, "wb") as f:
            pickle.dump(attr_names, f)

        exists = []
        for split in ["train", "valid"]:
            for f in ["images", "attrs"]:
                exists.append((colored_mnist_dir / split / f"{f}.npy").exists())

        if not all(exists):
            self.generate_data(root_dir, colored_mnist_dir, skewed_ratio, severity)

        train_imgs = np.load(colored_mnist_dir / "train" / "images.npy")
        test_imgs = np.load(colored_mnist_dir / "valid" / "images.npy")

        train_attrs = np.load(colored_mnist_dir / "train" / "attrs.npy")
        test_attrs = np.load(colored_mnist_dir / "valid" / "attrs.npy")

        self._y_array = np.concatenate([train_attrs[:, 0], test_attrs[:, 0], test_attrs[:, 0]])
        self._y_array = torch.from_numpy(self._y_array).long()
        self._y_size = 1
        self._n_classes = len(torch.unique(self._y_array))

        corruption_label = np.concatenate([train_attrs[:, 1], test_attrs[:, 1], test_attrs[:, 1]])
        corruption_label = torch.from_numpy(corruption_label).long()

        self._metadata_array = np.vstack([train_attrs, test_attrs, test_attrs])
        self._metadata_array = torch.from_numpy(self._metadata_array).long()
        self._metadata_fields = ["color_label", "y"]

        self._metadata_map = {
            "color_label": [f"c{i}" for i in range(1, 11)],  # Padding for str formatting
            "y": [" 0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        }
        self._original_resolution = (28, 28)
        self._input_array = np.vstack([train_imgs, test_imgs, test_imgs])

        self._split_scheme = "official"
        self._split_array = np.zeros_like(self._y_array)
        # dummy val set
        self._split_array = np.array(
            [0] * len(train_attrs) + [1] * len(test_attrs) + [2] * len(test_attrs),
            dtype=np.int64,
        )

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["color_label", "y"]))
        super().__init__(root_dir, download, self._split_scheme)

    def generate_data(self, root_dir, colored_mnist_dir, skewed_ratio, severity):
        protocol = COLORED_MNIST_PROTOCOL

        for split in ["train", "valid"]:
            dataset = MNIST(root_dir, train=(split == "train"), download=True)
            os.makedirs(os.path.join(colored_mnist_dir, split), exist_ok=True)

            if split == "train":
                bias_aligned_ratio = 1 - skewed_ratio
            else:
                bias_aligned_ratio = 0.1

            color_labels = make_attr_labels(torch.LongTensor(dataset.targets), bias_aligned_ratio)

            images, attrs = [], []
            for img, target_label, color_label in tqdm(
                zip(dataset.data, dataset.targets, color_labels),
                total=len(color_labels),
            ):
                colored_img = protocol[color_label.item()](img, severity)
                colored_img = np.moveaxis(np.uint8(colored_img), 0, 2)

                images.append(colored_img)
                attrs.append([target_label, color_label])

            image_path = os.path.join(colored_mnist_dir, split, "images.npy")
            np.save(image_path, np.array(images).astype(np.uint8))
            attr_path = os.path.join(colored_mnist_dir, split, "attrs.npy")
            np.save(attr_path, np.array(attrs).astype(np.uint8))

    def get_input(self, idx):
        img = self._input_array[idx]
        img = Image.fromarray(img)
        return img


class ColoredMNIST(DatasetBase):
    skewed_ratio: float
    severity: int

    def create(self):
        return CMNISTBase(
            download=True,
            root_dir=self.root,
            seed=self.seed,
            skewed_ratio=self.skewed_ratio,
            severity=self.severity,
        )

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
