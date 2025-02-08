import math

import pytest

from dataset import CombinatorialGrouper
from utils.al_dataset import ActiveLearningDataset


@pytest.fixture()
def al_dataset(dataset):
    return ActiveLearningDataset(dataset.train)


def test_labelled_get_array(al_dataset: ActiveLearningDataset):
    al_dataset.label_randomly(10)

    metadata_array = al_dataset.labelled_metadata_array
    y_array = al_dataset.labelled_y_array
    assert len(metadata_array) == 10
    assert len(y_array) == 10
    assert metadata_array.shape == (10, 3)
    assert y_array.shape == (10,)


def test_unlabelled_metadata_array(al_dataset: ActiveLearningDataset):
    al_dataset.label_randomly(10)
    metadata_array = al_dataset.unlabelled_metadata_array
    y_array = al_dataset.unlabelled_y_array
    assert len(metadata_array) == (len(al_dataset._dataset) - 10)
    assert len(y_array) == (len(al_dataset._dataset) - 10)
    assert metadata_array.shape == (len(al_dataset._dataset) - 10, 3)
    assert y_array.shape == (len(al_dataset._dataset) - 10,)


def test_grouper_setter(al_dataset: ActiveLearningDataset, dataset):
    al_dataset.set_grouper(dataset.attr_grouper, dataset.grouper)

    assert isinstance(al_dataset._attr_grouper, CombinatorialGrouper)
    assert isinstance(al_dataset._group_grouper, CombinatorialGrouper)


def test_labelled_stats(al_dataset: ActiveLearningDataset, dataset):
    al_dataset.set_grouper(dataset.attr_grouper, dataset.grouper)
    al_dataset.label_randomly(50)

    labelled_stats = al_dataset.labelled_stats
    assert labelled_stats["attr"].sum() == 50
    assert labelled_stats["group"].sum() == 50
    assert labelled_stats["class"].sum() == 50

    unlabelled_stats = al_dataset.unlabelled_stats
    assert unlabelled_stats["attr"].sum() == (len(al_dataset._dataset) - 50)
    assert unlabelled_stats["group"].sum() == (len(al_dataset._dataset) - 50)
    assert unlabelled_stats["class"].sum() == (len(al_dataset._dataset) - 50)


@pytest.mark.parametrize("n", [1, 10, 30])
def test_label_randomly_int(n, request):
    al_dataset: ActiveLearningDataset = request.getfixturevalue("al_dataset")
    al_dataset.label_randomly(n)
    len(al_dataset) == n


@pytest.mark.parametrize("n", [0.1, 0.5, 1 / 3, 0.99])
def test_label_randomly_ratio(n, request):
    al_dataset: ActiveLearningDataset = request.getfixturevalue("al_dataset")
    al_dataset.label_randomly(n)
    len(al_dataset) == math.floor(len(al_dataset._dataset) * n)


@pytest.mark.parametrize("n", [1, 40, 80])
def test_get_group_array(n, request):
    al_dataset: ActiveLearningDataset = request.getfixturevalue("al_dataset")
    dataset = request.getfixturevalue("dataset")
    al_dataset.set_grouper(dataset.attr_grouper, dataset.grouper)
    al_dataset.label_randomly(n)

    labelled, unlabelled = al_dataset.group_array
    assert len(labelled) == n
    assert len(unlabelled) == len(al_dataset._dataset) - n
    assert labelled.shape == (n,)
    assert unlabelled.shape == (len(al_dataset._dataset) - n,)
