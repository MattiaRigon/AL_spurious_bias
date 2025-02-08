import logging
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from shutil import make_archive

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image

logger = logging.getLogger("wandb_artifact")
api = wandb.Api()


def upload(artifact_type):
    def wrapper(fn):
        @wraps(fn)
        def log(self, name, *args, **kwargs):
            dir = fn(self, name, *args, **kwargs)
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(dir)
            wandb.log_artifact(artifact)
            logger.info(f"logging artifact \033[4m\033[1m{name}\033[0m ...")

        return log

    return wrapper


@dataclass
class Artifact:
    root: Path = field(default=Path("."))

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        if not self.root.exists():
            self.root.mkdir(parents=True)

    @upload("Folder")
    def folder(self, name, folder: Path) -> Path:
        make_archive(name, "zip", root_dir=str(folder))
        return folder.parent / f"{name}.zip"

    @upload("Array")
    def array(self, name, obj) -> Path:
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        dir = self.root / f"{name}.npy"
        with open(dir, "wb") as f:
            np.save(f, np.array(obj))
        return dir

    @upload("DataFrame")
    def df(self, name, data):
        # directly pass DataFrame to data
        if isinstance(data, pd.DataFrame):
            ...
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        else:
            raise ValueError
        data.to_csv(dir := (self.root / f"{name}.csv"), index=False)
        return dir

    @upload("Text")
    def txt(self, name, string):
        assert isinstance(string, str)
        with open(dir := (self.root / f"{name}.txt"), "w") as f:
            f.write(string)
        return dir

    @upload("Image")
    def image(self, name, image):
        assert isinstance(image, Image.Image)
        image.save(dir := (self.root / f"{name}.png"))
        return dir
