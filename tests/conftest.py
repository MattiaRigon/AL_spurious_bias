import logging
import random
from collections import OrderedDict
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from environs import Env
from torch.utils.data import DataLoader
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset

from dataset import CombinatorialGrouper, DatasetBase
from job import WandBConfig, WandBModeEnum
from model import ModelBase
from model.optim import CosineAnnealingLRConfig, SGDConfig
from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig


def pytest_addoption(parser):
    parser.addoption("--show_log", action="store_true")


@pytest.fixture(autouse=True)
def no_multiprocessing(mocker):
    mocker.patch("torch.multiprocessing.set_sharing_strategy")


def pytest_configure(config):
    torch.manual_seed(Env().int("SEED"))
    logging.getLogger("model").propagate = False
    logging.getLogger("wandb_artifact").propagate = False
    if not config.getoption("--show_log"):
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).propagate = False


@pytest.fixture
def train_cfg():
    cfg = TrainConfig(n_steps=50, batch_size=8, checkpoint_freq=5, workers=0)
    return cfg


@pytest.fixture
def wandb_cfg():
    return WandBConfig(WandBModeEnum.disabled, "", "", None)


@pytest.fixture
def test_cfg():
    cfg = TestConfig(batch_size=10, workers=0)
    return cfg


@pytest.fixture
def optimizer_cfg():
    return SGDConfig(1e-3, 0.9, 0)


@pytest.fixture
def lr_scheduler_cfg():
    return CosineAnnealingLRConfig(verbose=False, T_max=4, eta_min=0)


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 4)
        self.fc = nn.Linear(4, 2)

    def __call__(self, x, return_embedding=False):
        if not return_embedding:
            return self.fc(self.encoder(x))
        embed = self.encoder(x)
        return embed, self.fc(embed)


class Model(ModelBase):
    def __post_init__(self):
        super().__post_init__()
        self.model = SimpleModel()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def setup(self, *args, **kwargs): ...


@pytest.fixture
def model(optimizer_cfg, lr_scheduler_cfg):
    return Model("", optimizer_cfg, lr_scheduler_cfg)


@pytest.fixture
def wrapper_model(model, optimizer_cfg, lr_scheduler_cfg):
    wrapper = ModelWrapper(model, nn.CrossEntropyLoss())
    wrapper._get_checkpoints = Mock(
        return_value=OrderedDict(
            sorted({i: Model("", optimizer_cfg, lr_scheduler_cfg).state_dict() for i in range(10)}.items())
        )
    )
    return wrapper


def wild_subset(n):
    m = MagicMock(spec=WILDSSubset)
    m.__getitem__ = Mock(
        side_effect=lambda x: (
            torch.rand(8),
            random.randint(0, 1),
            torch.randint(0, 2, (3,)),
        )
    )
    m.__len__.return_value = n
    m.y_array = torch.randint(0, 2, (len(m),))
    m.metadata_array = torch.randint(0, 2, (len(m), 2))
    m.dataset = Mock(spec=WILDSDataset)
    m.dataset.n_classes = 2
    m._n_classes = 2
    m.transform = None
    return m


@pytest.fixture
def test_loader(test_subset):
    return DataLoader(test_subset, 10)


@pytest.fixture
def wild_dataset(mocker, rng) -> WILDSDataset:
    mocker.patch("os.path.exists")
    torch.manual_seed(Env().int("SEED"))

    class Dummy(WILDSDataset):
        def __init__(self):
            self._dataset_name = "dummy_dataset"
            self._data_dir = ""
            self._y_array = torch.randint(0, 2, (1000,))
            self._y_size = 1
            self._n_classes = 2
            self._metadata_array = torch.stack((torch.randint(0, 2, (len(self._y_array),)), self._y_array), dim=1)
            self._metadata_fields = ["s", "y"]
            self._metadata_map = {
                "s": ["a", "b"],
                "y": ["0", "1"],
            }
            self._split_array = torch.randint(0, 3, (len(self._y_array),))
            self._split_scheme = ""
            self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=(["s", "y"]))
            super().__init__("", False, None)

        def get_input(self, idx):
            return torch.rand(8)

    return Dummy()


@pytest.fixture
def dataset(wild_dataset):

    class Dummy(DatasetBase):
        def create(self):
            return wild_dataset

        def get_transform(self, _):
            return None

    return Dummy("", Env().int("SEED"), "")


@pytest.fixture
def train_subset(dataset):
    return dataset.train


@pytest.fixture
def test_subset(dataset):
    return dataset.test


@pytest.fixture
def rng():
    return np.random.RandomState(Env().int("SEED"))
