"""Tests for nkululeko/data/dataset.py -- Dataset.map_labels().

Regression: DATA.labels combined with a per-database DATA.<db>.filter can
collapse a split to a single remaining class (see issue #420 follow-up),
which previously only surfaced later as a cryptic xgboost
"base_score must be in (0,1)" error deep inside model training.
map_labels() must instead raise a readable NkululukoError right away.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nkululeko.data.dataset import Dataset
from nkululeko.utils.errors import NkululukoError


def _make_dataset(df, labels_config):
    ds = Dataset.__new__(Dataset)
    ds.name = "mydb"
    ds.context = type("Ctx", (), {"config": {"DATA": {"target": "emotion"}}})()

    util = MagicMock()
    util.exp_is_classification.return_value = True
    util.config_val.side_effect = lambda section, key, default: (
        labels_config if key == "labels" else default
    )
    util.config_val_data.side_effect = lambda name, key, default: default

    def _error(message):
        raise NkululukoError(f"ERROR: test: {message}")

    util.error.side_effect = _error
    ds.util = util
    return ds


class TestMapLabelsSingleClassGuard:
    def test_raises_when_filter_and_labels_mismatch_leaves_one_class(self):
        # emulates emodb.filter=['anger','happiness'] + labels=['neutral','anger']:
        # after both restrictions only 'anger' rows remain.
        df = pd.DataFrame({"emotion": ["anger", "anger", "happiness", "happiness"]})
        ds = _make_dataset(df, "['neutral', 'anger']")

        with pytest.raises(NkululukoError, match="only one class"):
            ds.map_labels(df)

    def test_passes_when_both_configured_classes_present(self):
        df = pd.DataFrame({"emotion": ["anger", "neutral", "anger", "neutral"]})
        ds = _make_dataset(df, "['neutral', 'anger']")

        result = ds.map_labels(df)
        assert set(result["emotion"].unique()) == {"anger", "neutral"}

    def test_no_labels_config_does_not_raise_for_naturally_single_class_data(self):
        # Without an explicit DATA.labels, a single-class dataframe is taken
        # at face value -- there's no configured expectation to contradict.
        df = pd.DataFrame({"emotion": ["anger", "anger", "anger"]})
        ds = _make_dataset(df, False)

        result = ds.map_labels(df)
        assert len(result) == 3
