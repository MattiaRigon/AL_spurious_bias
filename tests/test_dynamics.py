from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from environs import Env
from scipy.special import softmax

from job.dynamics import LearningDynamics
from utils.distance import Norm

from .test_distance import dist_obj  # noqa: F401


@pytest.fixture
def n_initial():
    return 0.1


@pytest.fixture
def ld_obj(n_initial, dataset, model, wandb_cfg, train_cfg, test_cfg, mocker):  # noqa: F811
    mocker.patch("wandb.log")
    mocker.patch("wandb.save")
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")
    mocker.patch("pandas.DataFrame.to_csv")
    mocker.patch("pathlib.Path.mkdir")

    job = LearningDynamics(
        n_initial=n_initial,
        seed=Env().int("SEED"),
        wandb=wandb_cfg,
        distance=Norm(),
        dataset=dataset,
        model=model,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )
    job._artifact.folder = Mock()
    job._artifact.array = Mock()
    return job


def test_compute_dist_matrix(ld_obj: LearningDynamics, dist_obj):  # noqa: F811
    n, k = 100, 5
    predictions = softmax(np.random.rand(n, 2, k), 1)
    m = ld_obj._compute_dist_matrix(predictions, dist_obj)
    assert m.shape == (k, k, n)


def test_learning_dynamics(ld_obj, model, mocker):
    mocker.patch("job.dynamics.open")
    torch_load = mocker.patch("torch.load")
    torch_load.return_value = ld_obj.wrapped_model.state_dict()
    ld_obj.wrapped_model._get_checkpoints = Mock(return_value={i: Path(f"model_step={i}.pkl") for i in range(10)})

    ld_obj.run()
    ld_obj._artifact.folder.call_count == (4 + 2 + 2) + 1


def test_eval_checkpoints(ld_obj, mocker):
    logger = mocker.patch("wandb.log")
    dataset = ld_obj.dataset.test
    steps = list(range(10))
    preds = np.random.randn(len(dataset), 2, len(steps))
    ld_obj._eval_checkpoints(dataset, preds, steps, "test")
    assert logger.call_count == len(steps)


def test_sample_and_save_disagree_matrix(ld_obj, mocker):
    opener = mocker.patch("job.dynamics.open")
    np_save = mocker.patch("job.dynamics.np.save")

    dataset = ld_obj.dataset.test
    folder = ld_obj._sample_and_save_disagree_matrix(dataset, np.random.randn(5, 5, len(dataset)), "aa", 17)
    assert opener.call_count == 4 + 2 + 2
    assert np_save.call_count == 4 + 2 + 2
    assert folder == Path("disagree_matrix_individual")
