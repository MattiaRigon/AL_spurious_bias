from abc import ABC, abstractclassmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing_extensions import dataclass_transform


@dataclass
@dataclass_transform()
class Distance(ABC):
    name: str

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls, eq=False)

    @abstractclassmethod
    def f(self, x: torch.TensorType, y: torch.TensorType):
        ...

    def __call__(self, x: torch.TensorType, y: torch.TensorType):
        return torch.vmap(self.f, in_dims=(0, 0))(x, y)


class Norm(Distance):
    name: str = "norm"

    def f(self, x, y):
        return torch.linalg.norm(x - y)


class KL_Divergence(Distance):
    name: str = "kl_divergence"

    def f(self, x, y):
        return F.kl_div(x.log(), y, reduction="none").sum()


class JS_Divergence(Distance):
    name: str = "js_divergence"

    @staticmethod
    def _f(x, y):
        return F.kl_div(x.log(), y, reduction="none").sum()

    def f(self, x, y):
        return 0.5 * self._f(x, y) + 0.5 * self._f(y, x)
