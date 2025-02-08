import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase

logger = logging.getLogger(__name__)


def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logger.info("Generating metadata for MetaShift...")
    dirs = {
        "train/cat/cat(indoor)": [1, 1],
        "train/dog/dog(outdoor)": [0, 0],
        "test/cat/cat(outdoor)": [1, 0],
        "test/dog/dog(indoor)": [0, 1],
    }

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(data_path, "MetaShift-Cat-Dog-indoor-outdoor", dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob("*.jpg"):
            all_data.append({"filename": img_path, "y": y, "place": g})
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(np.arange(len(df)), size=int(len(df) * test_pct), replace=False)
    val_idxs = rng.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size=int(len(df) * val_pct), replace=False)

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df["split"] = split_array.astype(int)
    df.to_csv(os.path.join(data_path, "metadata.csv"), index=False)


class MetaShiftBase(WILDSDataset):
    _dataset_name = "metashift"
    _versions_dict = {
        "1.0": {
            "download_url": "https://www.dropbox.com/s/a7k65rlj4ownyr2/metashift.tar.gz?dl=1",
            "compressed_size": None,
        }
    }

    def __init__(self, version=None, root_dir="data", download=False, split_scheme="official"):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not (version_file := Path(self._data_dir) / "RELEASE_v1.0.txt").exists():
            f = open(version_file, "a")
            f.close()

        if not os.path.exists(metafile := os.path.join(self.data_dir, "metadata.csv")):
            generate_metadata_metashift(self._data_dir)

        metadata_df = pd.read_csv(metafile)

        # Get the y values
        self._y_array = torch.LongTensor(metadata_df["y"].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack((torch.LongTensor(metadata_df["place"].values), self._y_array), dim=1)
        self._metadata_fields = ["background", "y"]
        self._metadata_map = {
            "background": [" outdoor", "indoor"],  # Padding for str formatting
            "y": [" dog", "cat"],
        }

        self._input_array = metadata_df["filename"].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = metadata_df["split"].values

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["background", "y"]))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = self._input_array[idx]
        x = Image.open(img_filename).convert("RGB")
        return x


class MetaShift(DatasetBase):
    def create(self):
        return MetaShiftBase(download=True, root_dir=self.root)

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
