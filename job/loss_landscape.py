import copy
from dataclasses import asdict, dataclass
from enum import Enum, auto
from functools import partial

import flatdict
import loss_landscapes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Subset

from dataset import WILDSSubset
from utils.modelwrapper import ModelWrapper, TestConfig, TrainConfig
from utils.wandb_artifact import Artifact

from . import JobBase


class Normalization(Enum):
    filter = auto()
    layer = auto()
    model = auto()


@dataclass
class RandomPlaneConfig:
    normalization: Normalization
    distance: int
    steps: int
    deepcopy_model: bool
    dict = asdict

    def __post_init__(self):
        self.normalization = self.normalization.name


@dataclass
class PlanarInterpolationConfig:
    steps: int
    deepcopy_model: bool
    dict = asdict


class PredictionProb(loss_landscapes.metrics.Metric):
    def __init__(self, x):
        super().__init__()
        self.x = torch.unsqueeze(x, 0)
        self.sign = {0: -1, 1: 1}

    def __call__(self, model):
        prob = torch.softmax(model.forward(self.x), dim=-1).squeeze()
        pred_label = torch.argmax(prob).item()
        return self.sign[pred_label] * prob[pred_label].item()


class Entropy(loss_landscapes.metrics.Metric):
    def __init__(self, x):
        super().__init__()
        self.x = torch.unsqueeze(x, 0)
        self.sign = {0: -1, 1: 1}

    def __call__(self, model):
        prob = torch.softmax(model.forward(self.x), dim=-1).squeeze()
        pred_label = torch.argmax(prob).item()
        ent = torch.sum(-prob * torch.log(prob))
        return self.sign[pred_label] * ent.item()


def plot_heatmap(x, *args, **kwargs):
    plt.close()
    fig, ax = plt.subplots()
    im = ax.imshow(x, *args, **kwargs)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.colorbar(im)
    return fig


class LossLandscape(JobBase):
    train_cfg: TrainConfig
    test_cfg: TestConfig
    landscape_cfg: PlanarInterpolationConfig
    num_samples_per_group: int

    def __post_init__(self):
        super().__post_init__()
        self.criterion = nn.CrossEntropyLoss()
        self.wrapped_model = ModelWrapper(self.model, self.criterion)
        self.artifact = Artifact()
        self.rng = np.random.RandomState(self.seed)
        self.eval_fn = partial(
            self.wrapped_model.eval_on_dataset,
            num_labels=self.dataset.n_classes,
            attr_grouper=self.dataset.attr_grouper,
            grouper=self.dataset.grouper,
            **self.test_cfg.dict(),
        )

    def get_loss_metric(self, size: int, dataset: WILDSSubset):
        g = self.dataset.grouper.metadata_to_group(dataset.metadata_array)

        def get_datapoints(ind):
            def stack(x, y):
                x = torch.vstack(x)
                if y[0].ndim == 1:
                    y = torch.concat(y)
                else:
                    y = torch.vstack(y)
                return x, y

            subset = Subset(dataset, ind)
            loader = DataLoader(subset, batch_size=64, shuffle=False)

            p = [batch for batch in loader]
            x, y, *_ = list(zip(*p))
            x, y = stack(x, y)
            return x, y

        metrics = []
        for i in torch.unique(g):
            ind = torch.where(g == i)[0]
            idx = self.rng.choice(ind, size=size, replace=False)
            x, y = get_datapoints(idx)
            m = loss_landscapes.metrics.Loss(self.criterion, x, y)
            metrics.append(m)
        return metrics

    def get_predict_metric(self, dataset: WILDSSubset, metric: loss_landscapes.metrics.Metric, upload_img=True):
        g = self.dataset.grouper.metadata_to_group(dataset.metadata_array)
        subset_wo_transform = copy.deepcopy(dataset)
        subset_wo_transform.transform = None
        metrics = []
        for i in torch.unique(g):
            ind = torch.where(g == i)[0]
            idx = self.rng.choice(ind, size=1, replace=False)[0]
            x, *_ = dataset[idx]
            metrics.append(metric(x))
            img, *_ = subset_wo_transform[idx]
            if isinstance(img, Image.Image):
                if upload_img:
                    self.artifact.image(f"group_{i}", img)
                    wandb.log({f"image/group_{i}": wandb.Image(img)})
        return metrics

    def get_end_models(self):
        ckpts = self.wrapped_model._get_checkpoints()
        ckpt_steps = list(ckpts.keys())
        s1, s2 = ckpt_steps[1], ckpt_steps[2]

        self.wrapped_model.model.load_state_dict(ckpts[s1])
        m1 = copy.deepcopy(self.wrapped_model.model).cpu().eval()
        self.wrapped_model.model.load_state_dict(ckpts[s2])
        m2 = copy.deepcopy(self.wrapped_model.model).cpu().eval()
        return m1, m2

    def run(self):
        model_initial = copy.deepcopy(self.wrapped_model.model)
        self.wrapped_model.train_on_dataset_epoch(self.dataset.train, **self.train_cfg.dict())

        test_metrics = self.eval_fn(self.dataset.test)
        test_metrics = dict(flatdict.FlatDict({"test": test_metrics}, delimiter="/"))
        wandb.log(test_metrics)

        loss_metrics = self.get_loss_metric(self.num_samples_per_group, self.dataset.train)
        pred_metrics = self.get_predict_metric(self.dataset.test, PredictionProb)
        ent_metrics = self.get_predict_metric(self.dataset.test, Entropy, upload_img=False)
        metrics = loss_metrics + pred_metrics + ent_metrics
        metric = loss_landscapes.metrics.MetricPipeline(metrics)

        model_initial.cpu()
        model_initial.eval()
        m1, m2 = self.get_end_models()

        result = loss_landscapes.planar_interpolation(model_initial, m1, m2, metric, **self.landscape_cfg.dict())
        for g in range(len(loss_metrics)):
            self.artifact.array(f"group_{g}_loss_map", result[..., g])
            fig = plot_heatmap(result[..., g])
            wandb.log({f"loss_map/group_{g}": fig})

        for g in range(len(pred_metrics)):
            i = g + len(loss_metrics)
            self.artifact.array(f"group_{g}_prediction_map", result[..., g])
            fig = plot_heatmap(result[..., g], vmin=-1, vmax=1, cmap="seismic")
            wandb.log({f"prediction_map/group_{g}": fig})

        for g in range(len(ent_metrics)):
            i = g + len(pred_metrics) + len(pred_metrics)
            self.artifact.array(f"group_{g}_entropy_map", result[..., i])
            fig = plot_heatmap(result[..., g], vmin=-1, vmax=1, cmap="seismic")
            wandb.log({f"entropy_map/group_{g}": fig})
