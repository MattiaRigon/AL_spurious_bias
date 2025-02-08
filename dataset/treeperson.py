from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
import wget
from PIL import Image
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

from . import DatasetBase


class TreePersonBase(WILDSDataset):
    _dataset_name = "treeperson"
    _versions_dict = {
        "1.0": {
            "download_url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
            "compressed_size": None,
            "metadata_url": "https://raw.githubusercontent.com/alextamkin/active-learning-pretrained-models/main/utils/datasets/treeperson/metadata.csv",  # noqa E501
        },
    }

    def __init__(self, root_dir, download, split_scheme="official"):
        self._version = None
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        f = open(self._data_dir / "RELEASE_v1.0.txt", "a")
        f.close()

        if not (self._data_dir / "metadata.csv").exists():
            wget.download(url=self._versions_dict["1.0"]["metadata_url"], out=str(self._data_dir))

        metadata_df = pd.read_csv(self.data_dir / "metadata.csv")
        # Get the y values
        self._y_array = torch.LongTensor(metadata_df["y"].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack((torch.LongTensor(metadata_df["place"].values), self._y_array), dim=1)
        self._metadata_fields = ["place", "y"]
        self._metadata_map = {
            "place": [" trees", "buildings"],  # Padding for str formatting
            "y": [" has no person", "has person"],
        }

        self._input_array = metadata_df["img_filename"].values
        self._original_resolution = (224, 224)

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = metadata_df["split"].values

        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["place", "y"]))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        img_filename = self.data_dir / self._input_array[idx]
        x = Image.open(img_filename).convert("RGB")
        return x

    def get_subset(self, split, frac=1, transform=None):
        if split == "test":
            split = "val"
        return super().get_subset(split, frac, transform)


class TreePerson(DatasetBase):
    def create(self):
        return TreePersonBase(download=True, root_dir=self.root)

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
