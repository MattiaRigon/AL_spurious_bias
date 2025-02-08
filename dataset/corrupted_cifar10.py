import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase
from .corrupted_cifar10_protocol import CORRUPTED_CIFAR10_PROTOCOL


def make_attr_labels(target_labels, bias_aligned_ratio):
    num_classes = target_labels.max().item() + 1
    num_samples_per_class = np.array([torch.sum(target_labels == label).item() for label in range(num_classes)])
    ratios_per_class = bias_aligned_ratio * np.eye(num_classes) + (1 - bias_aligned_ratio) / (num_classes - 1) * (
        1 - np.eye(num_classes)
    )

    corruption_milestones_per_class = (
        num_samples_per_class[:, np.newaxis] * np.cumsum(ratios_per_class, axis=1)
    ).round()
    num_corruptions_per_class = np.concatenate(  # noqa: F841
        [
            corruption_milestones_per_class[:, 0, np.newaxis],
            np.diff(corruption_milestones_per_class, axis=1),
        ],
        axis=1,
    )

    attr_labels = torch.zeros_like(target_labels)
    for label in range(10):
        indices = (target_labels == label).nonzero().squeeze()
        corruption_milestones = corruption_milestones_per_class[label]
        for corruption_idx, idx in enumerate(indices):
            attr_labels[idx] = np.min(np.nonzero(corruption_milestones > corruption_idx)[0]).item()

    return attr_labels


class CorruptedCifar10Base(WILDSDataset):
    _dataset_name = "corrupted_cifar10"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir: Path, download, seed, skewed_ratio, corruption_names, severity, postfix="0"):
        corrupted_cifar10_dir = root_dir / f"CorruptedCIFAR10-Type{postfix}-Skewed{skewed_ratio}-Severity{severity}"
        os.makedirs(corrupted_cifar10_dir, exist_ok=True)

        self._data_dir = corrupted_cifar10_dir
        f = open(Path(self._data_dir) / "RELEASE_v1.0.txt", "a")
        f.close()

        attr_names = ["object", "corruption"]
        attr_names_path = corrupted_cifar10_dir / "attr_names.pkl"
        with open(attr_names_path, "wb") as f:
            pickle.dump(attr_names, f)

        exists = []
        for split in ["train", "valid"]:
            for f in ["images", "attrs"]:
                exists.append((corrupted_cifar10_dir / split / f"{f}.npy").exists())

        if not all(exists):
            self.generate_data(root_dir, corrupted_cifar10_dir, skewed_ratio, corruption_names, severity)

        train_imgs = np.load(corrupted_cifar10_dir / "train" / "images.npy")
        val_imgs_ = np.load(corrupted_cifar10_dir / "valid" / "images.npy")

        train_attrs = np.load(corrupted_cifar10_dir / "train" / "attrs.npy")
        val_attrs_ = np.load(corrupted_cifar10_dir / "valid" / "attrs.npy")

        rng = np.random.RandomState(42)
        size_val = int(0.5 * len(val_attrs_))
        indexes = rng.permutation(len(val_attrs_))
        val_idx = indexes[:size_val]
        test_idx = indexes[size_val:]

        test_imgs = val_imgs_[test_idx]
        val_imgs = val_imgs_[val_idx]

        test_attrs = val_attrs_[test_idx]
        val_attrs = val_attrs_[val_idx]

        self._y_array = np.concatenate([train_attrs[:, 0], val_attrs[:, 0], test_attrs[:, 0]])
        self._y_array = torch.from_numpy(self._y_array).long()
        self._y_size = 1
        self._n_classes = len(torch.unique(self._y_array))

        corruption_label = np.concatenate([train_attrs[:, 1], val_attrs[:, 1], test_attrs[:, 1]])
        corruption_label = torch.from_numpy(corruption_label).long()

        self._metadata_array = np.vstack([train_attrs, val_attrs, test_attrs])
        self._metadata_array = torch.from_numpy(self._metadata_array).long()
        self._metadata_fields = ["corruption_label", "y"]
        self._metadata_map = {
            "corruption_label": corruption_names,  # Padding for str formatting
            "y": [" airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"],
        }
        self._original_resolution = (32, 32)
        self._input_array = np.vstack([train_imgs, val_imgs, test_imgs])

        self._split_scheme = "official"
        self._split_array = np.zeros_like(self._y_array)
        self._split_array = np.array(
            [0] * len(train_attrs) + [1] * len(val_attrs) + [2] * len(test_attrs),
            dtype=np.int64,
        )

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["corruption_label", "y"]))
        super().__init__(root_dir, download, self._split_scheme)

    def generate_data(self, root_dir, corrupted_cifar10_dir, skewed_ratio, corruption_names, severity):
        protocol = CORRUPTED_CIFAR10_PROTOCOL
        convert_img = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])

        for split in ["train", "valid"]:
            dataset = CIFAR10(root_dir, train=(split == "train"), download=True)
            os.makedirs(os.path.join(corrupted_cifar10_dir, split), exist_ok=True)

            if split == "train":
                bias_aligned_ratio = 1 - skewed_ratio
            else:
                bias_aligned_ratio = 0.1

            corruption_labels = make_attr_labels(torch.LongTensor(dataset.targets), bias_aligned_ratio)

            images, attrs = [], []
            for img, target_label, corruption_label in tqdm(
                zip(dataset.data, dataset.targets, corruption_labels),
                total=len(corruption_labels),
            ):
                method_name = corruption_names[corruption_label]
                corrupted_img = protocol[method_name](convert_img(img), severity + 1)
                images.append(np.array(corrupted_img).astype(np.uint8))
                attrs.append([target_label, corruption_label])

            image_path = os.path.join(corrupted_cifar10_dir, split, "images.npy")
            np.save(image_path, np.array(images).astype(np.uint8))
            attr_path = os.path.join(corrupted_cifar10_dir, split, "attrs.npy")
            np.save(attr_path, np.array(attrs).astype(np.uint8))

    def get_input(self, idx):
        img = self._input_array[idx]
        img = Image.fromarray(img)
        return img


class CorruptedCIFAR10(DatasetBase):
    skewed_ratio: float
    corruption_names: list[str]
    severity: int
    postfix: str

    def create(self):
        return CorruptedCifar10Base(
            download=True,
            root_dir=self.root,
            seed=self.seed,
            skewed_ratio=self.skewed_ratio,
            corruption_names=self.corruption_names,
            severity=self.severity,
            postfix=self.postfix,
        )

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
