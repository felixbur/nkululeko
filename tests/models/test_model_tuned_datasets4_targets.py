"""Regression for GH #437: TunedModel.train() (nkululeko/models/model_tuned.py:637)
does `targets = pd.DataFrame(self.dataset["train"]["targets"])`. With
datasets>=4, that double subscript returns a lazy `datasets.Column` object
(no `.dtype` attribute) instead of a plain list, and pd.DataFrame() can't
consume it directly:

    AttributeError: 'Column' object has no attribute 'dtype'

wrapping with list() forces it to a plain in-memory list first, which
pd.DataFrame() (and the .value_counts() call on it at model_tuned.py:644)
handle exactly as before. This test exercises the real `datasets` library
(not a mock), since the bug is specific to its Column type, not something a
mock would reproduce.
"""

import datasets
import pandas as pd
import pytest


class TestDatasetsColumnToDataFrame:
    def _make_targets_column(self):
        ds = datasets.Dataset.from_dict({"targets": [0, 1, 2, 1, 0]})
        return ds["targets"]

    def test_bare_column_is_not_directly_usable_by_pd_dataframe(self):
        # Documents the actual failure mode this bug produced, so this test
        # would have failed before the fix (and flags loudly if a future
        # datasets release changes Column's pd.DataFrame() compatibility,
        # rather than silently testing something no longer true).
        #
        # nkululeko supports datasets>=2.0.0, and this double-subscript only
        # returns the lazy Column type (no .dtype attribute) on datasets>=4;
        # on older, still-supported releases it's already a plain list, and
        # pd.DataFrame() on a plain list doesn't raise. Gate on what's
        # actually returned here, not on parsing datasets.__version__, since
        # that's the exact thing that determines whether this reproduces.
        column = self._make_targets_column()
        if isinstance(column, list):
            pytest.skip(
                "this datasets version already returns a plain list here; "
                "nothing to reproduce (see test_list_wrapped_column_builds_"
                "the_expected_dataframe for the version-portable fix check)"
            )
        with pytest.raises(AttributeError):
            pd.DataFrame(column)

    def test_list_wrapped_column_builds_the_expected_dataframe(self):
        column = self._make_targets_column()
        targets = pd.DataFrame(list(column))

        assert list(targets[0]) == [0, 1, 2, 1, 0]
        # Matches model_tuned.py:644's downstream usage for class weights.
        counts = targets[0].value_counts().sort_index()
        assert counts.to_dict() == {0: 2, 1: 2, 2: 1}
