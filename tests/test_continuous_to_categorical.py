import configparser
import tempfile
import numpy as np
import pytest

import nkululeko.glob_conf as glob_conf


def _make_config(labels=None, bins=None):
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test")
    config.set("EXP", "root", tempfile.gettempdir())
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")
    config.add_section("DATA")
    config.set("DATA", "target", "valence")
    if labels is not None:
        config.set("DATA", "labels", str(labels))
    if bins is not None:
        config.set("DATA", "bins", str(bins))
    config.add_section("MODEL")
    config.set("MODEL", "type", "xgb")
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")
    config.add_section("PLOT")
    config.add_section("REPORT")
    return config


def _make_reporter(truths, preds, config):
    glob_conf.init_config(config)
    from nkululeko.reporting.reporter import Reporter
    return Reporter(truths, preds, run=0, epoch=0)


class TestContinuousToCategoricalExplicitBins:
    def test_two_explicit_bins(self):
        config = _make_config(labels=["low", "high"], bins=[-np.inf, 0.5, np.inf])
        r = _make_reporter([0.2, 0.8, 0.4, 0.6], [0.1, 0.9, 0.3, 0.7], config)
        r.continuous_to_categorical()
        np.testing.assert_array_equal(r.truths, [0, 1, 0, 1])
        np.testing.assert_array_equal(r.preds, [0, 1, 0, 1])

    def test_three_explicit_bins(self):
        config = _make_config(
            labels=["low", "mid", "high"],
            bins=[-np.inf, 0.33, 0.66, np.inf],
        )
        r = _make_reporter([0.1, 0.5, 0.9], [0.2, 0.4, 0.8], config)
        r.continuous_to_categorical()
        np.testing.assert_array_equal(r.truths, [0, 1, 2])
        np.testing.assert_array_equal(r.preds, [0, 1, 2])

    def test_idempotent(self):
        config = _make_config(labels=["low", "high"], bins=[-np.inf, 0.5, np.inf])
        r = _make_reporter([0.2, 0.8], [0.1, 0.9], config)
        r.continuous_to_categorical()
        truths_after_first = r.truths.copy()
        r.continuous_to_categorical()  # second call must be a no-op
        np.testing.assert_array_equal(r.truths, truths_after_first)


class TestContinuousToCategoricalAutoEquidistant:
    def test_two_labels_midpoint_split(self):
        """With 2 labels and uniform data, cut should be at the midpoint."""
        config = _make_config(labels=["low", "high"])
        # uniform data 0..1; p5=0.05, p95=0.95, midpoint~0.5
        data = np.linspace(0, 1, 100)
        r = _make_reporter(data, data.copy(), config)
        r.continuous_to_categorical()
        assert set(r.truths).issubset({0, 1})
        # lower half → 0, upper half → 1
        assert r.truths[0] == 0
        assert r.truths[-1] == 1

    def test_three_labels_equidistant(self):
        """With 3 labels, bin edges split the p5–p95 range into thirds."""
        config = _make_config(labels=["low", "mid", "high"])
        data = np.linspace(0, 1, 200)
        r = _make_reporter(data, data.copy(), config)
        r.continuous_to_categorical()
        assert set(r.truths).issubset({0, 1, 2})
        assert r.truths[0] == 0
        assert r.truths[-1] == 2

    def test_outliers_do_not_shift_bins(self):
        """Values far outside p5–p95 range should land in the outer bins, not alter edges."""
        config = _make_config(labels=["low", "high"])
        core = np.linspace(0, 1, 90)
        outliers = np.array([-100.0, -50.0, 50.0, 100.0, 200.0])
        data = np.concatenate([core, outliers])
        r = _make_reporter(data, data.copy(), config)
        r.continuous_to_categorical()
        # All values must be assigned to a valid bin index
        assert set(r.truths).issubset({0, 1})
        # Large positive outliers → bin 1 (high)
        assert r.truths[-1] == 1
        # Large negative outliers → bin 0 (low)
        assert r.truths[len(core)] == 0

    def test_default_labels_applied_when_missing(self):
        """When no labels are set, defaults ['low','high'] are used → 2 bins."""
        config = _make_config()  # no labels, no bins
        data = np.linspace(0, 1, 50)
        r = _make_reporter(data, data.copy(), config)
        r.continuous_to_categorical()
        assert set(r.truths).issubset({0, 1})
