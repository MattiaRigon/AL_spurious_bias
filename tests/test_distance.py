import pytest
import torch

from utils import distance
from utils.misc import to_prob


@pytest.fixture(
    params=[
        distance.Norm(),
        distance.KL_Divergence(),
        distance.JS_Divergence(),
    ],
    ids=lambda x: x.name,
)
def dist_obj(request):
    return request.param


def test_distance(dist_obj):
    x, y = torch.randn(100, 2), torch.randn(100, 2)
    x, y = to_prob(x), to_prob(y)
    dist = dist_obj(x, y)
    assert torch.all(torch.gt(dist, 0))
