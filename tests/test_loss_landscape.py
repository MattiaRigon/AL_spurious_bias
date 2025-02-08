from collections import OrderedDict
from unittest.mock import Mock

import pytest
from environs import Env

import job.loss_landscape
from job.loss_landscape import LossLandscape, PlanarInterpolationConfig


@pytest.fixture
def loss_landscape(wandb_cfg, dataset, train_cfg, test_cfg, model, mocker):
    mocker.patch("wandb.save")
    mocker.patch("wandb.log")
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")
    mocker.patch("wandb.log_artifact")
    mocker.patch("utils.wandb_artifact.open")
    mocker.patch("utils.wandb_artifact.np.save")
    mocker.patch("wandb.Artifact.add_file")
    mocker.patch("job.loss_landscape.plot_heatmap")

    obj = LossLandscape(
        seed=Env().int("SEED"),
        dataset=dataset,
        wandb=wandb_cfg,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        model=model,
        landscape_cfg=PlanarInterpolationConfig(10, True),
        num_samples_per_group=50,
    )
    obj.wrapped_model._get_checkpoints = Mock(return_value=OrderedDict({i: model.state_dict() for i in range(5)}))
    return obj


def test_loss_landscape(loss_landscape):
    loss_landscape.run()


def test_get_loss_metric(loss_landscape, test_subset):
    metrics = loss_landscape.get_loss_metric(10, test_subset)
    assert len(metrics) == 4
    for m in metrics:
        assert len(m.inputs) == 10
        assert len(m.target) == 10


@pytest.mark.parametrize("metric", [job.loss_landscape.PredictionProb, job.loss_landscape.Entropy])
@pytest.mark.parametrize("upload_img", [False, True])
def test_get_predict_metric(loss_landscape, test_subset, metric, upload_img):
    metrics = loss_landscape.get_predict_metric(test_subset, metric, upload_img=upload_img)
    assert len(metrics) == 4


def test_PredictionProb_metric(loss_landscape, test_subset):
    from job.loss_landscape import PredictionProb

    model = loss_landscape.wrapped_model.model
    model.to(device="cpu")
    model.forward = lambda x: loss_landscape.wrapped_model.model(x)
    for i in range(10):
        x, y, *_ = test_subset[i]
        m = PredictionProb(x)
        out = m(model)
        assert -1 <= out <= 1


def test_Entropy_metric(loss_landscape, test_subset):
    from job.loss_landscape import Entropy

    model = loss_landscape.wrapped_model.model
    model.to(device="cpu")
    model.forward = lambda x: loss_landscape.wrapped_model.model(x)
    for i in range(10):
        x, y, *_ = test_subset[i]
        m = Entropy(x)
        out = m(model)
        assert -1 <= out <= 1
