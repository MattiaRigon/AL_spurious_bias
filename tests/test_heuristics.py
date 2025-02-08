import numpy as np
from environs import Env

from utils import heuristics
from utils.distance import Norm


def test_CoreSet():
    train_embeddings = np.random.rand(100, 128)
    unlabelled_embeddings = np.random.rand(1000, 128)
    heuristic = heuristics.CoreSet()
    heuristic.add_labelled_stats(train_embeddings)
    heuristic.get_ranks(unlabelled_embeddings, 5)
    assert len(heuristic.labelled_embeddings) == 105


def test_VariationRatio():
    model_output = np.random.rand(100, 3, 10)
    heuristic = heuristics.VariationRatio()
    scores = heuristic.compute_score(model_output)
    assert len(scores) == 100


def test_ReversedMargin():
    model_output = np.random.rand(100, 2, 10)
    heuristic = heuristics.ReversedMargin()
    scores = heuristic.get_uncertainties(model_output)
    assert len(scores) == 100


def test_QBCxMargin():
    model_output = np.random.rand(100, 2, 10)
    heuristic = heuristics.QBCxMargin(distance=Norm(), qbc_weight=0.5)
    scores = heuristic.get_uncertainties([model_output] * 2)
    assert len(scores[0]) == 100
    assert len(scores[1]) == 100


def test_TwoStageHeuristic():
    model_output = np.random.rand(100, 2, 10)
    heuristic = heuristics.TwoStageHeuristic(
        heuristic1=heuristics.Random(),
        heuristic2=heuristics.Entropy(),
        first_batch_ratio=1.5,
    )

    to_label, uncertainty = heuristic.get_ranks(model_output, 10)
    assert len(to_label) == len(uncertainty) == 10


def test_ClusterMargin_build_hac():
    embeddings = np.random.rand(300, 256)
    heuristic = heuristics.ClusterMargin(Env().int("SEED"), 10, "average")
    heuristic.build_hac(embeddings)
    assert len(heuristic.HAC_list.labels_) == 300


def test_ClusterMargin_margin_data():
    heuristic = heuristics.ClusterMargin(Env().int("SEED"), 10, "average")
    predictions = np.random.rand(300, 2)
    idx = heuristic.margin_data(predictions)
    assert len(idx) == 300


def test_ClusterMargin_get_ranks():
    rng = np.random.RandomState(0)
    embeddings = rng.rand(n := 1000, 256)
    label_idx = rng.permutation(len(embeddings))[: (k := 50)]
    labelled_map = np.zeros(len(embeddings))
    labelled_map[label_idx] = 1
    labelled_map = labelled_map.astype(bool)

    heuristic = heuristics.ClusterMargin(Env().int("SEED"), 10, "average")
    heuristic.add_labelled_map(labelled_map)
    heuristic.build_hac(embeddings)

    predictions = rng.rand(n - k, 2, 1)
    to_label, _ = heuristic.get_ranks(predictions, 50)

    def _pool_to_oracle_index(index):
        lbl_nz = (~labelled_map).nonzero()[0]
        return np.array([int(lbl_nz[idx].squeeze().item()) for idx in index])

    for idx in _pool_to_oracle_index(to_label):
        assert labelled_map[idx] == 0
