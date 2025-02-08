import math

from baal.active import ActiveLearningDataset as ALD
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.utils import get_counts
from wilds.datasets.wilds_dataset import WILDSSubset


class ActiveLearningDataset(ALD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self._dataset, WILDSSubset)

    def set_grouper(self, attr_grouper: CombinatorialGrouper, group_grouper: CombinatorialGrouper):
        self._attr_grouper = attr_grouper
        self._group_grouper = group_grouper

    def _attr_stats(self, meta_array):
        return self._attr_grouper.metadata_to_group(meta_array, return_counts=True)[1]

    def _group_stats(self, meta_array):
        return self._group_grouper.metadata_to_group(meta_array, return_counts=True)[1]

    def _get_stats(self, meta, y):
        return {
            "attr": self._attr_stats(meta),
            "group": self._group_stats(meta),
            "class": get_counts(y, self._dataset._n_classes),
        }

    def label_randomly(self, n: int = 1) -> None:
        assert n > 0, f"n must be > 0 but received {n}"
        if isinstance(n, int):
            return super().label_randomly(n)
        elif isinstance(n, float):
            assert 0 < n < 1, f"n must be in (0,1) but received {n}"
            n = math.floor(n * self.n_unlabelled)
            return super().label_randomly(n)

    @property
    def labelled_stats(self):
        return self._get_stats(self.labelled_metadata_array, self.labelled_y_array)

    @property
    def unlabelled_stats(self):
        return self._get_stats(self.unlabelled_metadata_array, self.unlabelled_y_array)

    @property
    def labelled_metadata_array(self):
        assert hasattr(self._dataset, "metadata_array")
        return self._dataset.metadata_array[self.labelled.nonzero()[0]]

    @property
    def labelled_y_array(self):
        assert hasattr(self._dataset, "y_array")
        return self._dataset.y_array[self.labelled.nonzero()[0]]

    @property
    def unlabelled_metadata_array(self):
        assert hasattr(self._dataset, "metadata_array")
        return self._dataset.metadata_array[(~self.labelled).nonzero()[0]]

    @property
    def unlabelled_y_array(self):
        assert hasattr(self._dataset, "y_array")
        return self._dataset.y_array[(~self.labelled).nonzero()[0]]

    @property
    def group_array(self):
        labelled = self._group_grouper.metadata_to_group(self.labelled_metadata_array)
        unlabelled = self._group_grouper.metadata_to_group(self.unlabelled_metadata_array)
        return labelled, unlabelled
