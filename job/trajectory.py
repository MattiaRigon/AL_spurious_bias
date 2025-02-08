import copy
from functools import partial

import flatdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader

from dataset import WILDSSubset
from utils.landscape import Loss, Metric, MetricPipeline, PredictionProb, SurfaceOutcome, loss_landscape
from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig, to_device
from utils.wandb_artifact import Artifact

from . import JobBase


def plot_heatmap(surface: SurfaceOutcome, key: int, *args, **kwargs):
    plt.close()
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    coords_x, coords_y = surface.coords
    ct = ax.contourf(coords_x, coords_y, surface.metrics[key].matrix, levels=35, alpha=0.9, *args, **kwargs)
    ax.plot(surface.path_2d[:, 0], surface.path_2d[:, 1], color="k", lw=1)
    ax.plot(surface.path_2d[0][0], surface.path_2d[0][1], "ko")
    ax.plot(surface.path_2d[-1][0], surface.path_2d[-1][1], "kx")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.colorbar(ct)
    return fig


def trajectory2df(path_2d: np.ndarray):
    x, y = path_2d[:, 0], path_2d[:, 1]
    return pd.DataFrame({"x": x, "y": y})


def surface2df(coords: tuple[np.ndarray, np.ndarray], matrix: np.ndarray):
    coords_x, coords_y = coords
    x, y, val = [], [], []
    for i in range(len(coords_x)):
        for j in range(len(coords_y)):
            x.append(coords_x[i])
            y.append(coords_y[j])
            val.append(matrix[j, i])
    return pd.DataFrame({"x": x, "y": y, "v": val})


class Trajectory(JobBase):
    train_cfg: TrainConfig
    test_cfg: TestConfig
    resolution: int
    subset_size: int

    def __post_init__(self):
        super().__post_init__()
        self.criterion = nn.CrossEntropyLoss()
        self.wrapped_model = ModelWrapper(self.model, self.criterion)
        self.device = self.wrapped_model.device
        self.artifact = Artifact()
        self.rng = np.random.RandomState(self.seed)
        self.eval_fn = partial(
            self.wrapped_model.eval_on_dataset,
            num_labels=self.dataset.n_classes,
            attr_grouper=self.dataset.attr_grouper,
            grouper=self.dataset.grouper,
            **self.test_cfg.dict(),
        )

        loader = DataLoader(
            self.dataset.train,
            batch_size=self.subset_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        x, y, *_ = next(iter(loader))
        loss_metric = Loss(to_device(x, self.device), to_device(y, self.device), self.criterion)
        prob_metric = self.get_predict_metric(self.dataset.test, PredictionProb, True)
        self.metrics = MetricPipeline(
            {"loss": loss_metric, **{f"prob_g{i}": prob_metric[i] for i in range(len(prob_metric))}}
        )

    def run(self):
        self.wrapped_model.train_on_dataset_epoch(self.dataset.train, **self.train_cfg.dict())

        test_metrics = self.eval_fn(self.dataset.test)
        test_metrics = dict(flatdict.FlatDict({"test": test_metrics}, delimiter="/"))
        wandb.log(test_metrics)

        surface = loss_landscape(self.wrapped_model, self.metrics, self.resolution, self.seed)

        self.artifact.array("loss_map_mat", surface.metrics["loss"].matrix)
        self.artifact.df("loss_map", surface2df(surface.coords, surface.metrics["loss"].matrix))
        fig = plot_heatmap(surface, "loss", cmap="YlGnBu")
        wandb.log({"loss_map/overall": wandb.Image(fig)})

        for key in surface.metrics.keys():
            if "prob" not in key:
                continue
            self.artifact.array(f"{key}_map_mat", surface.metrics[key].matrix)
            self.artifact.df(f"{key}_map", surface2df(surface.coords, surface.metrics[key].matrix))
            fig = plot_heatmap(surface, key, cmap="seismic")
            wandb.log({f"prob_map/{key}": wandb.Image(fig)})

        self.artifact.df("trajectory", trajectory2df(surface.path_2d))
        self.artifact.array("x_coords_array", surface.coords[0])
        self.artifact.array("y_coords_array", surface.coords[1])
        self.artifact.array("trajectory_array", surface.path_2d)
        self.artifact.array("pcvariances", surface.pcvariances)
        self.artifact.array("loss_optim_point", np.array(surface.metrics["loss"].true_optim_point))

    def get_predict_metric(self, dataset: WILDSSubset, metric: Metric, upload_img=True):
        g = self.dataset.grouper.metadata_to_group(dataset.metadata_array)
        subset_wo_transform = copy.deepcopy(dataset)
        subset_wo_transform.transform = None
        metrics = []
        for i in torch.unique(g):
            ind = torch.where(g == i)[0]
            idx = self.rng.choice(ind, size=1, replace=False)[0]
            x, *_ = dataset[idx]
            metrics.append(metric(to_device(x, self.device)))
            img, *_ = subset_wo_transform[idx]
            if isinstance(img, Image.Image):
                if upload_img:
                    self.artifact.image(f"group_{i}", img)
                    wandb.log({f"image/group_{i}": wandb.Image(img)})
        return metrics
