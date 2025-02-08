from collections import OrderedDict
from unittest.mock import Mock

import pytest
from environs import Env

from job.trajectory import Trajectory


@pytest.fixture
def trajectory(wandb_cfg, dataset, train_cfg, test_cfg, model, mocker):
    mocker.patch("wandb.save")
    mocker.patch("wandb.log")
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")
    mocker.patch("wandb.log_artifact")
    mocker.patch("utils.wandb_artifact.open")
    mocker.patch("utils.wandb_artifact.np.save")
    mocker.patch("wandb.Artifact.add_file")
    mocker.patch("job.loss_landscape.plot_heatmap")
    mocker.patch("pandas.DataFrame.to_csv")

    train_cfg.checkpoint_freq = 1
    train_cfg.n_steps = 10

    obj = Trajectory(
        seed=Env().int("SEED"),
        dataset=dataset,
        wandb=wandb_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        model=model,
        subset_size=100,
        resolution=10,
    )
    obj.wrapped_model._get_checkpoints = Mock(return_value=OrderedDict({i: model.state_dict() for i in range(10)}))
    return obj


def test_trajectory(trajectory):
    trajectory.run()
