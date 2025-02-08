import logging
from functools import partial
from utils.dataloader import InfiniteDataLoader
from tqdm import tqdm
import flatdict
import torch
import torch.nn as nn
import wandb

from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig

from . import JobBase

logger = logging.getLogger(__name__)


class ERM(JobBase):
    train_cfg: TrainConfig
    test_cfg: TestConfig

    def __post_init__(self):
        super().__post_init__()
        self.wrapped_model = ModelWrapper(self.model, nn.CrossEntropyLoss())
        self.eval_fn = partial(
            self.wrapped_model.eval_on_dataset,
            num_labels=self.dataset.n_classes,
            attr_grouper=self.dataset.attr_grouper,
            grouper=self.dataset.grouper,
            **self.test_cfg.dict(),
        )

    def get_split_stats(self):
        def return_dataset_stats(D):
            def return_counts(array):
                c, p = {}, {}
                for a, b in zip(*torch.unique(array, return_counts=True)):
                    c[str(int(a))] = int(b)
                    p[str(int(a))] = int(b) / len(array)
                return c, p

            m = {}
            for t, (c, p) in [
                ("class", return_counts(D.y_array)),
                ("attr", return_counts(self.dataset.attr_grouper.metadata_to_group(D.metadata_array))),
                ("group", return_counts(self.dataset.grouper.metadata_to_group(D.metadata_array))),
            ]:
                m[t] = c
                m[f"{t}_ratio"] = p
            return m

        stats = {}
        for s in ["train", "test", "val"]:
            stats[s] = return_dataset_stats(getattr(self.dataset, s))
        return dict(flatdict.FlatDict(stats, delimiter="/"))

    def run(self):
        logger.info(
            f"Starting training: n_steps={self.train_cfg.n_steps}, checkpoint_freq={self.train_cfg.checkpoint_freq}"
            + f", dataset={len(self.dataset.train)}"  # noqa: W503
        )

        optimizer, lr_scheduler = self.model.get_optimizer()
        loader = iter(
            InfiniteDataLoader(
                self.dataset.train,
                None,
                self.train_cfg.batch_size,
                num_workers=self.train_cfg.workers,
            )
        )
        for step in tqdm(range(self.train_cfg.n_steps)):
            self.wrapped_model.train()
            data, target, *_ = next(loader)
            _ = self.wrapped_model.train_on_batch(data, target, optimizer, lr_scheduler, None)
            if (step % self.train_cfg.checkpoint_freq == 0) or (step == self.train_cfg.n_steps - 1):
                test_metrics, val_metrics = self.eval_fn(self.dataset.test), self.eval_fn(self.dataset.val)
                metrics = dict(flatdict.FlatDict({"test": test_metrics, "val": val_metrics}, delimiter="/"))
                wandb.log({**metrics, "train/step": step})
