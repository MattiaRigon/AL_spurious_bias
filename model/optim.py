from copy import deepcopy
from dataclasses import dataclass, field

from torch.optim import SGD  # noqa: F401, E402
from torch.optim import Adam  # noqa: F401, E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: F401, E402
from typing_extensions import dataclass_transform


@dataclass
@dataclass_transform()
class OptimizerConfig:
    _target_: str = ""

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def __str__(self):
        name = self.__class__.__name__
        name = name.replace("Config", "")
        params = deepcopy(self.__dict__)
        if "_target_" in params:
            del params["_target_"]
        return f"{name}({', '.join([f'{f}={value}' for f, value in params.items()])})"


class SGDConfig(OptimizerConfig):
    lr: float
    momentum: float
    weight_decay: float
    _target_: str = field(default="model.optim.SGD", init=False)


class AdamConfig(OptimizerConfig):
    lr: float
    weight_decay: float
    _target_: str = field(default="model.optim.Adam", init=False)


@dataclass
@dataclass_transform()
class SchedulerConfig:
    verbose: bool
    _target_: str = ""

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(cls)

    def __str__(self):
        name = self.__class__.__name__
        name = name.replace("Config", "")
        params = deepcopy(self.__dict__)
        if "_target_" in params:
            del params["_target_"]
        return f"{name}({', '.join([f'{f}={value}' for f, value in params.items()])})"


class CosineAnnealingLRConfig(SchedulerConfig):
    T_max: int
    eta_min: float
    _target_: str = field(default="model.optim.CosineAnnealingLR", init=False)
