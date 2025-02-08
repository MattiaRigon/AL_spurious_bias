import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .modelwrapper import ModelWrapper


def get_init_from_flat_params_fn(model):
    shapes = []
    for name, param in model.named_parameters():
        shapes.append((name, param.shape, param.numel()))

    def _unflatten_to_state_dict(flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter : counter + tnum].reshape(tsize)  # noqa: E203
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict

    def fn(model, flat_params):
        if not isinstance(flat_params, torch.Tensor):
            raise AttributeError("Argument to init_from_flat_params() must be torch.Tensor")
        state_dict = _unflatten_to_state_dict(flat_params, shapes)
        model.load_state_dict(state_dict, strict=True)
        return model

    return fn


def get_flat_params(model, state_dict, device):
    """Get flattened and concatenated params of the model."""
    m = copy.deepcopy(model)
    m.load_state_dict(state_dict)
    params = {}
    for name, param in m.named_parameters():
        params[name] = param.data

    flat_params = torch.Tensor()
    flat_params = flat_params.to(device)
    # if torch.cuda.is_available():
    #     flat_params = flat_params.cuda()
    # if torch.backends.mps.is_available():
    #     flat_params = flat_params.to(torch.device("mps"))

    for _, param in params.items():
        flat_params = torch.cat((flat_params, torch.flatten(param)))
    return flat_params


@dataclass
class MetricOutcome:
    matrix: np.ndarray
    extreme_coord: tuple[int, int]
    extreme_val: float
    true_optim_point: tuple[float, float] = field(default=None)


class Metric(ABC):
    def __init__(self, track="min") -> None:
        self.matrix = []
        assert track in ["min"]
        self.track = track
        if self.track == "min":
            self._extreme_val = float("inf")
            self._extreme_coord = None

    def compare(self, val, coord):
        if self.track == "min":
            if val < self._extreme_val:
                self._extreme_val = val
                self._extreme_coord = coord

    def get_matrix(self):
        return np.array(self.matrix).T

    @abstractmethod
    def get_value(self, model: Callable):
        pass

    def __call__(self, model: Callable, coord: tuple[int, int]):
        val = self.get_value(model)
        self.row.append(val)
        self.compare(val, coord)

    def init_row(self):
        self.row = []

    def end_row(self):
        self.matrix.append(self.row)
        self.init_row()

    @property
    def result(self):
        return MetricOutcome(self.get_matrix(), self._extreme_coord, self._extreme_val)


class Loss(Metric):
    def __init__(self, X, y, loss_fn) -> None:
        super().__init__()
        self.X, self.y, self.loss_fn = X, y, loss_fn

    def get_value(self, model):
        return self.loss_fn(model(self.X), self.y).cpu().item()


class PredictionProb(Metric):
    def __init__(self, x):
        super().__init__()
        self.x = torch.unsqueeze(x, 0)
        self.sign = {0: -1, 1: 1}

    def get_value(self, model):
        prob = torch.softmax(model(self.x), dim=-1).squeeze()
        pred_label = torch.argmax(prob).item()
        return self.sign[pred_label] * prob[pred_label].cpu().item()


class MetricPipeline:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        assert isinstance(metrics, dict)
        self.metrics = metrics

    def __call__(self, model: Callable, coord: tuple[int, int]):
        for _, m in self.metrics.items():
            m(model, coord)

    def init_row(self):
        for _, m in self.metrics.items():
            m.init_row()

    def end_row(self):
        for _, m in self.metrics.items():
            m.end_row()

    @property
    def result(self):
        result = {}
        for key, m in self.metrics.items():
            result[key] = m.result
        return result


@dataclass
class SurfaceOutcome:
    metrics: MetricOutcome | dict[str, MetricOutcome]
    coords: tuple[np.ndarray, np.ndarray]
    path_2d: np.ndarray
    pcvariances: np.ndarray


def loss_landscape(model_wrapper: ModelWrapper, metric: Metric, res: int, seed: int):
    from .loss_landscape import DimReduction, LossGrid

    device = model_wrapper.device
    model = model_wrapper.model
    ckpts = model_wrapper._get_checkpoints()
    assert len(ckpts) > 0

    optim_path = [get_flat_params(model, w, device) for _, w in ckpts.items()]
    dim_reduction = DimReduction(optim_path, "pca", seed)

    reduced_dict = dim_reduction.reduce()
    path_2d = reduced_dict["path_2d"]
    directions = reduced_dict["reduced_dirs"]
    pcvariances = reduced_dict.get("pcvariances")

    loss_grid = LossGrid(
        optim_path=optim_path,
        model=model,
        metric=metric,
        path_2d=path_2d,
        directions=directions,
        init_fn=get_init_from_flat_params_fn(model),
        res=res,
    )

    return SurfaceOutcome(loss_grid.surface, loss_grid.coords, path_2d, pcvariances)
