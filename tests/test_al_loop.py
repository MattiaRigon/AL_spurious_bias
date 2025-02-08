import math
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from environs import Env

from job.active import ActiveLearning
from utils import heuristics
from utils.distance import JS_Divergence, Norm


@pytest.fixture
def heuristic():
    return heuristics.RandomConfig()


@pytest.fixture
def query_size():
    return 4


@pytest.fixture
def save_acq_score():
    return False


@pytest.fixture
def n_al_steps():
    return 3


@pytest.fixture
def resume():
    return None


@pytest.fixture
def al_loop(
    dataset,
    heuristic,
    model,
    train_cfg,
    test_cfg,
    wandb_cfg,
    query_size,
    save_acq_score,
    n_al_steps,
    request,
    mocker,
    resume,
):
    mocker.patch("wandb.log")
    mocker.patch("wandb.save")
    mocker.patch("os.makedirs")
    mocker.patch("torch.save")
    mocker.patch("utils.wandb_artifact.open")
    mocker.patch("wandb.log_artifact")
    mocker.patch("wandb.Artifact.add_file")
    mocker.patch("utils.wandb_artifact.Artifact.folder")

    cache = request.config.cache
    tempfolder = Path(tempfile.mkdtemp(dir=cache._cachedir))
    ActiveLearning._acq_score_path = Mock(return_value=Path(tempfolder))

    if resume:
        labelled_map = np.zeros(len(dataset.train))
        for i in range(1, 6):
            idx = np.random.choice(np.where(labelled_map == 0)[0], 10, replace=False)
            labelled_map[idx] = i
        np.save(resumepath := (tempfolder / "labelled_map.npy"), labelled_map)
    else:
        resumepath = None

    obj = ActiveLearning(
        wandb=wandb_cfg,
        seed=Env().int("SEED"),
        dataset=dataset,
        model=model,
        heuristic=heuristic,
        reset_each_round=True,
        n_al_steps=n_al_steps,
        n_initial=30,
        query_size=query_size,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        save_dict=save_acq_score,
        resume=resumepath,
    )
    yield obj
    shutil.rmtree(tempfolder)


def test_al_loop_on_start(al_loop: ActiveLearning):
    al_loop.on_start()
    assert hasattr(al_loop, "_init_weights")
    assert hasattr(al_loop, "_init_al_state")


@pytest.mark.parametrize(
    "heuristic",
    [
        heuristics.MarginConfig(),
        heuristics.RandomConfig(),
        heuristics.BADGEConfig(Env().int("SEED")),
        heuristics.CertaintyConfig(),
        heuristics.EntropyConfig(),
        heuristics.OracleConfig(Env().int("SEED")),
        heuristics.QBCConfig(JS_Divergence()),
        heuristics.QBCRawConfig(Norm()),
        heuristics.IWQBCConfig(JS_Divergence(), 1.0),
        heuristics.QBCRandomConfig(Norm(), Env().int("SEED")),
        heuristics.CoreSetConfig(),
        heuristics.PowerMarginConfig(Env().int("SEED"), 0.0, 1.0),
        heuristics.PowerQBCConfig(Env().int("SEED"), 0.0, 1.0, JS_Divergence()),
        heuristics.QBCEntropyConfig(),
        heuristics.VariationRatioConfig(),
        heuristics.QBCBALDConfig(),
        heuristics.KMeansConfig(Env().int("SEED")),
        heuristics.QBCxMarginConfig(JS_Divergence(), 0.5),
        heuristics.TwoStageQBCxMarginConfig(JS_Divergence(), 2),
        heuristics.BAITConfig(0.5, 50),
        heuristics.ClusterMarginConfig(Env().int("SEED"), 10, "average"),
    ],
    ids=lambda x: x._target_.split(".")[-1],
)
def test_al_loop_run(al_loop: ActiveLearning, model, mocker):
    torch_load = mocker.patch("torch.load")
    torch_load.return_value = al_loop.wrapped_model.state_dict()

    k = 10
    al_loop.wrapped_model._get_checkpoints = Mock(return_value={i: Path(f"model_step={i}.pkl") for i in range(k)})

    assert len(al_loop.al_dataset) == al_loop.n_initial
    assert len(al_loop.al_dataset.pool) == len(al_loop.dataset.train) - al_loop.n_initial
    al_loop.run()
    assert len(al_loop.al_dataset) == al_loop.n_al_steps * al_loop.query_size + al_loop.n_initial
    assert len(al_loop.al_dataset.pool) == len(al_loop.dataset.train) - (
        al_loop.n_al_steps * al_loop.query_size + al_loop.n_initial
    )


def test_al_loop_run_no_reset(al_loop: ActiveLearning, model):
    al_loop.reset_each_round = False
    al_loop.wrapped_model._get_checkpoints = Mock(return_value={i: model.state_dict() for i in range(5)})
    al_loop.run()


@pytest.mark.parametrize("query_size", [0.1, 0.2])
def test_al_loop_run_query_ratio(al_loop: ActiveLearning, query_size, mocker, model):
    T = math.floor(query_size * len(al_loop.dataset.train))
    assert al_loop.query_size == T
    al_loop.run()
    assert len(al_loop.al_dataset) == (al_loop.n_al_steps * T) + al_loop.n_initial


@pytest.mark.parametrize("save_acq_score", [True])
def test_al_loop_save_scores(al_loop: ActiveLearning):
    al_loop.run()


@pytest.mark.parametrize("n_al_steps,query_size", [(100, 100)])
def test_al_loop_empty_unlabelled(al_loop: ActiveLearning):
    al_loop.run()


@pytest.mark.parametrize("resume", [True, None])
def test_al_loop_resume(al_loop: ActiveLearning):
    num_samples = 50 if al_loop.resume else 30
    assert len(al_loop.al_dataset) == num_samples
    if al_loop.resume:
        assert al_loop.al_dataset.current_al_step == 5
        al_loop.al_dataset.label_randomly(10)
        assert al_loop.al_dataset.current_al_step == 6
