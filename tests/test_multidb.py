"""Tests for nkululeko/multidb.py.

Regression for https://github.com/felixbur/nkululeko/issues/407: the
multidb heatmap and results.txt always labeled the metric "UAR", even for
regression experiments where the actual metric (MSE, MAE, CCC, ...) is
something else entirely.
"""

import configparser
from unittest.mock import MagicMock, patch

import numpy as np

from nkululeko.multidb import _metric_label, plot_heatmap


def _make_config(exp_type=None, measure=None):
    config = configparser.ConfigParser()
    config.add_section("EXP")
    if exp_type is not None:
        config["EXP"]["type"] = exp_type
    if measure is not None:
        config.add_section("MODEL")
        config["MODEL"]["measure"] = measure
    return config


class TestMetricLabel:
    def test_classification_default_is_uar(self):
        config = _make_config(exp_type="classification")
        assert _metric_label(config) == "UAR"

    def test_regression_default_is_mse(self):
        config = _make_config(exp_type="regression")
        assert _metric_label(config) == "MSE"

    def test_regression_custom_measure_mae(self):
        config = _make_config(exp_type="regression", measure="mae")
        assert _metric_label(config) == "MAE"

    def test_regression_custom_measure_ccc(self):
        config = _make_config(exp_type="regression", measure="ccc")
        assert _metric_label(config) == "CCC"

    def test_regression_custom_measure_pcc(self):
        config = _make_config(exp_type="regression", measure="pcc")
        assert _metric_label(config) == "PCC"

    def test_classification_custom_measure_eer(self):
        config = _make_config(exp_type="classification", measure="eer")
        assert _metric_label(config) == "EER"

    def test_missing_exp_type_defaults_to_classification(self):
        config = _make_config()
        assert _metric_label(config) == "UAR"

    def test_missing_exp_section_defaults_to_classification(self):
        config = configparser.ConfigParser()
        assert _metric_label(config) == "UAR"

    def test_missing_model_section_uses_default_measure(self):
        config = configparser.ConfigParser()
        config.add_section("EXP")
        config["EXP"]["type"] = "regression"
        assert _metric_label(config) == "MSE"


class TestPlotHeatmap:
    """Covers both outputs plot_heatmap produces: results.txt *and* the
    heatmap's own title/caption (ax.set_title()) -- the original bug report
    is specifically about the heatmap label, not just the text file.

    Plotting calls (plt.figure/savefig/close, sn.heatmap) are patched out:
    it keeps these tests fast and independent of a real matplotlib/seaborn
    rendering backend, while still letting us capture the exact caption
    string passed to ax.set_title().
    """

    def _run_plot_heatmap(self, config, tmp_path):
        results = np.array([[0.1, 0.2], [0.3, 0.4]])
        last_epochs = np.array([[1, 2], [3, 4]])
        mock_ax = MagicMock()
        with (
            patch("nkululeko.multidb.plt.figure"),
            patch("nkululeko.multidb.sn.heatmap", return_value=mock_ax),
            patch("nkululeko.multidb.plt.savefig"),
            patch("nkululeko.multidb.plt.close"),
        ):
            plot_heatmap(
                results,
                last_epochs,
                ["a", "b"],
                str(tmp_path / "heatmap.png"),
                config,
                ["a", "b"],
            )
        return mock_ax

    def test_results_txt_labels_regression_metric(self, tmp_path):
        config = _make_config(exp_type="regression", measure="mae")
        config["EXP"]["root"] = str(tmp_path)

        self._run_plot_heatmap(config, tmp_path)

        content = (tmp_path / "results.txt").read_text()
        assert "Mean MAE:" in content
        assert "Mean UAR:" not in content

    def test_heatmap_caption_labels_regression_metric(self, tmp_path):
        config = _make_config(exp_type="regression", measure="mae")
        config["EXP"]["root"] = str(tmp_path)

        mock_ax = self._run_plot_heatmap(config, tmp_path)

        caption = mock_ax.set_title.call_args.args[0]
        assert "Mean MAE:" in caption
        assert "UAR" not in caption

    def test_results_txt_still_labels_uar_for_classification(self, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)

        self._run_plot_heatmap(config, tmp_path)

        content = (tmp_path / "results.txt").read_text()
        assert "Mean UAR:" in content

    def test_heatmap_caption_still_labels_uar_for_classification(self, tmp_path):
        config = _make_config(exp_type="classification")
        config["EXP"]["root"] = str(tmp_path)

        mock_ax = self._run_plot_heatmap(config, tmp_path)

        caption = mock_ax.set_title.call_args.args[0]
        assert "Mean UAR:" in caption
