import logging
import os
from functools import partial
from pathlib import Path

import flatdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from baal.active.dataset.pytorch_dataset import ActiveLearningPool
from colorama import Fore, Style

from dataset import WILDSSubset
from utils.al_dataset import ActiveLearningDataset
from utils.distance import Distance
from utils.eval_helper import eval_metrics, process_predictions
from utils.misc import _to_torch
from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig
from utils.wandb_artifact import Artifact

from . import JobBase

logger = logging.getLogger(__name__)


class LearningDynamics(JobBase):
    n_initial: int | float
    distance: Distance
    train_cfg: TrainConfig
    test_cfg: TestConfig

    def __post_init__(self):
        super().__post_init__()
        self.al_dataset = ActiveLearningDataset(self.dataset.train, random_state=self.seed)
        self.wrapped_model = ModelWrapper(self.model, nn.CrossEntropyLoss())

        get_prob_fn = getattr(self.wrapped_model, "checkpoints_predictions_on_dataset")
        self._get_prob_fn = partial(get_prob_fn, **self.test_cfg.dict(), return_steps=True)
        self.al_dataset.label_randomly(self.n_initial)
        self._artifact = Artifact()

    @staticmethod
    def _compute_dist_matrix(preds, dist_fn) -> np.ndarray:
        logger.info("Compute distance matrix")

        preds = _to_torch(preds)
        dist_matrix = torch.vmap(torch.vmap(dist_fn, in_dims=(None, 2)), in_dims=(2, None))(preds, preds)
        return dist_matrix.numpy()

    def run(self):
        self.wrapped_model.train_on_dataset(self.al_dataset, **self.train_cfg.dict())
        matrix_path = Path("disagree_matrix")
        os.makedirs(matrix_path, exist_ok=True)

        def agg(dist_m: np.ndarray, T: torch.TensorType, name: str, split: str):
            def agg_and_save_matrix(m: np.ndarray, path: Path):
                for agg in [np.mean, np.std]:
                    df = pd.DataFrame(agg(m, -1))
                    save_path = path.parent / (path.name + f"_{agg.__name__}.csv")
                    df.to_csv(save_path)
                    logger.debug(f"save csv: {save_path}")

            os.makedirs(save_path := matrix_path / split / name, exist_ok=True)
            for t in T.unique():
                idx = np.where(T == t)[0]
                m = dist_m[..., idx]
                agg_and_save_matrix(m, (save_path / f"{t}").absolute())

        for split in ["val", "test", "unlabelled"]:
            logger.info(f"Running on {Style.BRIGHT}{Fore.RED}{split}{Style.RESET_ALL} set")
            if split == "unlabelled":
                dataset: ActiveLearningPool = self.al_dataset.pool
                setattr(dataset, "metadata_array", self.al_dataset.unlabelled_metadata_array)
                setattr(dataset, "y_array", self.al_dataset.unlabelled_y_array)
            else:
                dataset: WILDSSubset = getattr(self.dataset, split)

            steps, preds = self._get_prob_fn(dataset)
            self._eval_checkpoints(dataset, preds, steps, split)
            dist_m = self._compute_dist_matrix(preds, self.distance)
            indi_folder = self._sample_and_save_disagree_matrix(dataset, dist_m, split)

            agg(dist_m, self.dataset.grouper.metadata_to_group(dataset.metadata_array), "group", split)
            agg(dist_m, self.dataset.attr_grouper.metadata_to_group(dataset.metadata_array), "attr", split)
            agg(dist_m, dataset.y_array, "class", split)

        self._artifact.folder("disagree_matrix", matrix_path.absolute())
        self._artifact.folder("disagree_matrix_individual", indi_folder.absolute())

    def _sample_and_save_disagree_matrix(self, dataset, disagree_matrix: np.ndarray, split_name: str, num_samples=20):
        groups = self.dataset.grouper.metadata_to_group(dataset.metadata_array).numpy()
        attr = self.dataset.attr_grouper.metadata_to_group(dataset.metadata_array).numpy()
        target = dataset.y_array.numpy()

        folder = Path("disagree_matrix_individual") / split_name
        folder.mkdir(parents=True, exist_ok=True)

        def save(m: np.ndarray, idx: int, name: str):
            with open(folder / f"{name}{idx}.npy", "wb") as f:
                np.save(f, m)

        rng = np.random.RandomState(self.seed)
        for name, T in [("group", groups), ("attr", attr), ("class", target)]:
            for idx, indices in enumerate([np.where(T == t)[0] for t in np.unique(T)]):
                ind = rng.choice(indices, num_samples, replace=len(indices) < num_samples)
                save(disagree_matrix[..., ind], idx, name)
        return folder.parent

    def _eval_checkpoints(self, dataset, predictions, steps, name):
        groups = self.dataset.grouper.metadata_to_group(dataset.metadata_array).numpy()
        attr = self.dataset.attr_grouper.metadata_to_group(dataset.metadata_array).numpy()
        target = dataset.y_array.numpy()
        fn = partial(eval_metrics, targets=target, attributes=attr, gs=groups)

        for i, step in enumerate(steps):
            prob = process_predictions(predictions[..., i], self.dataset.n_classes)
            m = fn(preds=prob)
            m = dict(flatdict.FlatDict({name: m}, delimiter="/"))
            m["ckpt_step"] = step
            wandb.log(m)
