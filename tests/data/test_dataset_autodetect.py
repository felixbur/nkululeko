"""Tests for experiment-type auto-detection in nkululeko/data/dataset.py.

Covers the ``Dataset._autodetect_experiment_type`` helper that was extracted
from ``load()``: it must only set ``[EXP] type`` to ``regression`` when no
type is configured, respect (and warn about) an explicit non-regression type,
and do nothing for non-numeric labels.
"""

import configparser

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

import nkululeko.glob_conf as glob_conf
from nkululeko.data.dataset import Dataset


class _FakeUtil:
    """Minimal Util stand-in: real is_numeric + config_val + warn capture."""

    def __init__(self, config):
        self.config = config
        self.warnings = []

    def is_numeric(self, series):
        # Mirror the real Util.is_numeric (DataFrameMixin) behaviour.
        return is_numeric_dtype(series)

    def config_val(self, section, key, default):
        if self.config is None:
            return default
        try:
            return self.config[section][key]
        except KeyError:
            return default

    def warn(self, message):
        self.warnings.append(message)


@pytest.fixture(autouse=True)
def _reset_glob_conf():
    """Ensure glob_conf is clean between tests and reset afterwards."""
    glob_conf.config = None
    yield
    glob_conf.config = None


def _make_dataset(config, col_label="age"):
    """Create a bare Dataset wired to a fake util (no audformat db needed)."""
    # The helper reads via self.util.config_val and writes via glob_conf.config,
    # so both must reference the same configparser object (as in production).
    glob_conf.config = config
    ds = Dataset.__new__(Dataset)
    ds.name = "test_db"
    ds.col_label = col_label
    ds.util = _FakeUtil(config)
    return ds


def _numeric_df():
    return pd.DataFrame({"age": [23.0, 45.0, 60.0]})


def _categorical_df():
    return pd.DataFrame({"emotion": ["happy", "sad", "angry"]})


class TestAutodetectExperimentType:
    def test_sets_regression_when_no_type_configured(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        ds = _make_dataset(config)
        ds._autodetect_experiment_type(_numeric_df())
        assert glob_conf.config["EXP"]["type"] == "regression"
        assert ds.util.warnings == []

    def test_warns_and_respects_non_regression_type(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config.set("EXP", "type", "classification")
        ds = _make_dataset(config)
        ds._autodetect_experiment_type(_numeric_df())
        # User's configured type is respected, not overwritten.
        assert glob_conf.config["EXP"]["type"] == "classification"
        # A single warning is emitted that names the label column.
        assert len(ds.util.warnings) == 1
        msg = ds.util.warnings[0]
        assert "age" in msg
        assert "classification" in msg

    def test_no_warn_when_regression_already_configured(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config.set("EXP", "type", "regression")
        ds = _make_dataset(config)
        ds._autodetect_experiment_type(_numeric_df())
        assert glob_conf.config["EXP"]["type"] == "regression"
        assert ds.util.warnings == []

    def test_no_change_for_non_numeric_labels(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config.set("EXP", "type", "classification")
        ds = _make_dataset(config, col_label="emotion")
        ds._autodetect_experiment_type(_categorical_df())
        assert glob_conf.config["EXP"]["type"] == "classification"
        assert ds.util.warnings == []
