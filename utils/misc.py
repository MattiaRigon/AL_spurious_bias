# from https://github.com/baal-org/baal/blob/master/baal/utils/cuda_utils.py
import logging
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import partialmethod, singledispatch, wraps

import numpy as np
import torch
from tqdm import tqdm


@singledispatch
def to_device(data, device: torch.device):
    return data


@to_device.register(torch.Tensor)
@to_device.register(torch.nn.Module)
def _to_device_modules(data, device):
    return data.to(device)


@to_device.register
def _to_device_mapping(data: Mapping, device):
    # use the type of the object to create a new one:
    return type(data)([(key, to_device(val, device)) for key, val in data.items()])  # type: ignore


@to_device.register
def _to_device_sequence(data: Sequence, device):
    # use the type of this object to instantiate a new one:
    if hasattr(data, "_fields"):  # in case it's a named tuple
        return type(data)(*(to_device(item, device) for item in data))
    elif isinstance(data, str):
        # Special case
        return data
    else:
        return type(data)(to_device(item, device) for item in data)  # type: ignore


@singledispatch
def _to_torch(x):
    return x


@_to_torch.register(torch.Tensor)
def _to_torch_tensor(x):
    return x


@_to_torch.register(np.ndarray)
def _to_torch_np_array(x):
    return torch.from_numpy(x)


def to_torch(fn):
    @wraps(fn)
    def wrapper(probabilities):
        probabilities = _to_torch(probabilities)
        return fn(probabilities)

    return wrapper


def to_prob(probabilities: torch.TensorType):
    not_bounded = torch.min(probabilities) < 0 or torch.max(probabilities) > 1.0
    multiclass = probabilities.shape[1] > 1
    sum_to_one = torch.allclose(probabilities.sum(1), torch.tensor(1.0))
    if not_bounded or (multiclass and not sum_to_one):
        if multiclass:
            probabilities = torch.softmax(probabilities, 1)
        else:
            probabilities = torch.sigmoid(probabilities)
    return probabilities


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    from https://gist.github.com/simon-weber/7853144
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    logging.disable(highest_level)

    try:
        yield
    finally:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
        logging.disable(previous_level)
