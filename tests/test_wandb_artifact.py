from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import torch
import wandb

from utils.wandb_artifact import Artifact


@pytest.fixture
def artifact(monkeypatch):
    monkeypatch.setattr(wandb, "log_artifact", Mock())
    monkeypatch.setattr(wandb.Artifact, "add_file", Mock())
    art = Artifact()
    return art


def test_upload_folder(artifact):
    with patch("utils.wandb_artifact.make_archive") as make_archive:
        artifact.folder("test_path", Path("something"))
    assert make_archive.call_args == call("test_path", "zip", root_dir="something")


def test_upload_array(artifact, mocker):
    filename = "test_array"
    opener = mocker.patch("utils.wandb_artifact.open")
    np_save = mocker.patch("utils.wandb_artifact.np.save")

    artifact.array(filename, np.random.randn(5, 5))
    opener.assert_called_with(Path(".") / f"{filename}.npy", "wb")
    np_save.assert_called_once()
    np_save.reset_mock()

    artifact.array("test_array", torch.rand(5, 5))
    opener.assert_called_with(Path(".") / f"{filename}.npy", "wb")
    np_save.assert_called_once()


def test_upload_dataframe(artifact, mocker):
    filename = "test_df"
    saver = mocker.patch("utils.wandb_artifact.pd.DataFrame.to_csv")

    artifact.df(filename, pd.DataFrame([(1, 2), (3, 4)]))
    saver.assert_called_with(Path(".") / f"{filename}.csv", index=False)
    saver.assert_called_once()
    saver.reset_mock()

    artifact.df(filename, {"a": [1, 2], "b": [1, 2]})
    saver.assert_called_with(Path(".") / f"{filename}.csv", index=False)
    saver.assert_called_once()


def test_upload_text(artifact, mocker):
    filename = "test_txt"
    opener = mocker.patch("utils.wandb_artifact.open")

    artifact.txt(filename, "asdfasd")
    opener.assert_called_with(Path(".") / f"{filename}.txt", "w")
