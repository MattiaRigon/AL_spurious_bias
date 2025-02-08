import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase


class BarBase(WILDSDataset):
    _dataset_name = "bar"
    _versions_dict = {
        "1.0": {
            "download_url": "https://github.com/alinlab/BAR/archive/master.tar.gz",
            "compressed_size": None,
        }
    }

    def __init__(self, root_dir, download, split_scheme="official"):
        self._version = None
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        f = open(self._data_dir / "RELEASE_v1.0.txt", "a")
        f.close()

        data_folder = self._data_dir / "BAR-master"
        with open(data_folder / "metadata.json") as f:
            metadata = json.load(f)

        classes, filename, split_array = [], [], []
        folder = {True: "train", False: "test"}
        for key, val in metadata.items():
            classes.append(val["cls"])
            filename.append(str(data_folder / folder[val["train"]] / f"{key}.jpg"))
            split_array.append(0 if val["train"] else 2)

        # duplicate for dummy val
        for key, val in metadata.items():
            if not val["train"]:
                classes.append(val["cls"])
                filename.append(str(data_folder / folder[val["train"]] / f"{key}.jpg"))
                split_array.append(1)

        self._input_array = np.array(filename)
        self._original_resolution = (224, 224)

        le = LabelEncoder()
        self._y_array = torch.LongTensor(le.fit_transform(classes))
        self._y_size = 1
        self._n_classes = len(le.classes_)

        self._metadata_array = torch.stack((torch.zeros_like(self._y_array), self._y_array), dim=1)
        self._metadata_fields = ["dummy", "y"]
        self._metadata_map = {
            "dummy": [" "],  # Padding for str formatting
            "y": le.classes_,
        }

        self._split_scheme = split_scheme
        self._split_array = np.array(split_array)
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["dummy", "y"]))
        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = self._input_array[idx]
        x = Image.open(img_filename).convert("RGB")
        return x


class BAR(DatasetBase):
    def create(self):
        return BarBase(download=True, root_dir=self.root)

    def get_transform(self, split):
        if split in ["test", "val"]:
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif split == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise ValueError(f"{split} not found")