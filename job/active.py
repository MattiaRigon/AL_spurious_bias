import gc
import logging
import math
from copy import deepcopy
from functools import partial
from pathlib import Path

import flatdict
import numpy as np
import torch
import torch.nn as nn
import wandb
from colorama import Fore, Style
from hydra.utils import instantiate

from utils.active_loop import ActiveLearningLoop
from utils.al_dataset import ActiveLearningDataset
from utils.heuristics import (
    AbstractHeuristic,
    HeuristicsConfig,
    ImportanceWeighting,
    RequireLabelledStats,
    ClusterMargin,
)
from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig
from utils.wandb_artifact import Artifact
from utils.rrr_loss import RRRLoss 
from . import JobBase

logger = logging.getLogger(__name__)


class ActiveLearning(JobBase):
    n_al_steps: int
    n_initial: int | float
    query_size: int | float
    reset_each_round: bool
    heuristic: HeuristicsConfig
    train_cfg: TrainConfig
    test_cfg: TestConfig
    save_dict: bool
    resume: Path | None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.query_size, float):
            assert 0 < self.query_size < 1
            self.query_size = math.floor(self.query_size * len(self.dataset.train))

        if self.resume:
            labelled_map = np.load(self.resume)
            assert len(labelled_map) == len(self.dataset.train)

        self.al_dataset = ActiveLearningDataset(
            self.dataset.train,
            random_state=self.seed,
            labelled=labelled_map if self.resume else None,
            pool_specifics={"transform": self.dataset.get_transform("test")},
        )
        self.al_dataset.set_grouper(
            attr_grouper=self.dataset.attr_grouper,
            group_grouper=self.dataset.grouper,
        )

        # self.wrapped_model = ModelWrapper(self.model, nn.CrossEntropyLoss())
        self.wrapped_model = ModelWrapper(self.model, RRRLoss())

        get_prob_fn = getattr(self.wrapped_model, self.heuristic.get_prob_fn_name)
        get_prob_fn = partial(get_prob_fn, **self.test_cfg.dict())
        self.heuristic: AbstractHeuristic = instantiate(self.heuristic)
        self.heuristic.device = self.model.device

        if self.save_dict:
            save_acq_score_folder = self._acq_score_path()
            save_acq_score_folder.mkdir(exist_ok=True)
        else:
            save_acq_score_folder = None

        self.active_loop = ActiveLearningLoop(
            self.al_dataset,
            get_prob_fn,
            self.heuristic,
            query_size=self.query_size,
            uncertainty_folder=save_acq_score_folder,
        )

        if not self.resume:
            self.al_dataset.label_randomly(self.n_initial)

        if isinstance(self.heuristic, ClusterMargin):
            get_embed_fn = getattr(self.wrapped_model, "embedding_on_dataset")
            get_embed_fn = partial(get_embed_fn, **self.test_cfg.dict())
            embed = get_embed_fn(self.dataset.train, **self.active_loop.kwargs)
            self.heuristic.build_hac(embed)

        self.eval_fn = partial(
            self.wrapped_model.eval_on_dataset,
            num_labels=self.dataset.n_classes,
            attr_grouper=self.dataset.attr_grouper,
            grouper=self.dataset.grouper,
            **self.test_cfg.dict(),
        )

        if isinstance(self.heuristic, ImportanceWeighting):
            self._weights_map: dict[int, np.ndarray] = {}

        self._wandb_artifact = Artifact()

    def _acq_score_path(self):
        return Path(".") / "save_acq_score"

    def on_start(self):
        self._init_weights = deepcopy(self.model.state_dict())
        self._init_al_state = self.al_dataset.state_dict()

    def on_end(self):
        self._wandb_artifact.array("labelled_map", self.al_dataset.labelled_map)
        if self.save_dict:
            self._wandb_artifact.folder("al_state_dict", self._acq_score_path())

    def on_loop_start(self, step):
        torch.cuda.empty_cache()
        gc.collect()
        if self.reset_each_round:
            self.model.load_state_dict(self._init_weights)
        logger.info(
            f"{Fore.MAGENTA}AL step={step}/{self.n_al_steps}"
            f" labelled={len(self.al_dataset)}"
            f" unlabelled={len(self.al_dataset.pool)}"
            f" heuristic={self.heuristic.__class__.__name__}{Style.RESET_ALL}"
        )

        if isinstance(self.heuristic, ClusterMargin):
            self.heuristic.add_labelled_map(self.al_dataset.labelled)

    def log(self, metrics: dict, step: int):
        def group_counts(d: dict):
            new_d = {}
            for name, item in d.items():
                new_d[name] = {str(int(i)): int(count) for i, count in enumerate(item)}
            for name, item in d.items():
                new_d[f"{name}_ratio"] = {
                    str(int(i)): int(count) / item.sum().item() if item.sum().item() > 0 else 0
                    for i, count in enumerate(item)
                }
            return new_d

        metrics.update(
            {
                "AL": {
                    "step": step,
                    "label_to_total_ratio": len(self.al_dataset) / (len(self.al_dataset) + len(self.al_dataset.pool)),
                    "labelled": {
                        "total": len(self.al_dataset),
                        **group_counts(self.al_dataset.labelled_stats),
                    },
                    "unlabelled": {
                        "total": len(self.al_dataset.pool),
                        **group_counts(self.al_dataset.unlabelled_stats),
                    },
                }
            }
        )
        metrics = dict(flatdict.FlatDict(metrics, delimiter="/"))
        wandb.log(metrics)

    def _get_weights(self):
        labelled_map = self.al_dataset.labelled_map
        if (scores := self.heuristic.weights) is not None:
            step = self.al_dataset.labelled_map.max()
            idx = np.where((labelled_map == step) | (labelled_map == 0))[0]

            weights = np.ones(self.al_dataset.labelled.sum())
            self._weights_map[step] = scores[labelled_map[idx] == step]
            for i, w in self._weights_map.items():
                weights[labelled_map[labelled_map > 0] == i] = w
            return weights
        return None

    def run(self):
        self.on_start()
        for step in range(self.n_al_steps + 1):
            self.on_loop_start(step)
            if isinstance(self.heuristic, ImportanceWeighting):
                weights = self._get_weights()
            else:
                weights = None
            self.wrapped_model.train_on_dataset(self.al_dataset, weights=weights, **self.train_cfg.dict())

            # evaluate
            test_metrics = self.eval_fn(self.dataset.test)
            val_metrics = self.eval_fn(self.dataset.val)
            self.log({"test": test_metrics, "val": val_metrics}, step)

            # generate embeddings for coreset
            if isinstance(self.heuristic, RequireLabelledStats):
                labelled_info = self.active_loop.get_probabilities(self.al_dataset, **self.active_loop.kwargs)
                self.heuristic.add_labelled_stats(labelled_info)
                assert len(self.heuristic._labelled_stats) == len(self.al_dataset)

            # label
            if step < self.n_al_steps:
                if not self.active_loop.step():
                    break

        self.on_end()
