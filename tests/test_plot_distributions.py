"""Tests for plot_distributions statistical result storage in plots.py."""

import ast
import configparser
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    """Provide a minimal config and mock report for every test."""
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "testexp",
        "root": str(tmp_path),
    }
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    config["EXPL"] = {"value_counts": "[(['emotion', 'age'])]"}
    config["PLOT"] = {"format": "png", "titles": "False"}
    glob_conf.config = config
    glob_conf.report = MagicMock()
    yield
    glob_conf.config = None


def _make_df():
    """Categorical label + one continuous and one categorical attribute."""
    rng = np.random.default_rng(0)
    n = 60
    labels = pd.Categorical(["happy"] * 20 + ["sad"] * 20 + ["neutral"] * 20)
    age = rng.normal(loc=30, scale=5, size=n)  # continuous
    gender = pd.Categorical(["m"] * 30 + ["f"] * 30)  # categorical
    df = pd.DataFrame({"class_label": labels, "age": age, "gender": gender})
    df["class_label"] = df["class_label"].astype("category")
    return df


class TestPlotDistributionsStats:
    """Statistical results are written for categorical-label vs continuous-attribute."""

    def _make_plots(self, tmp_path):
        from nkululeko.plots import Plots

        fig_dir = os.path.join(str(tmp_path), "images")
        res_dir = os.path.join(str(tmp_path), "results", "run_0")
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(res_dir, exist_ok=True)

        with patch(
            "nkululeko.utils.util.Util.get_path",
            side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir + "/",
        ):
            plots = Plots()
        return plots, res_dir

    def test_result_file_created(self, tmp_path):
        """A result file is created for a categorical-label vs continuous-attribute pair."""
        plots, _ = self._make_plots(tmp_path)
        df = _make_df()
        glob_conf.config["EXPL"]["value_counts"] = "[['age']]"

        fig_dir = os.path.join(str(tmp_path), "images")
        res_dir_path = os.path.join(str(tmp_path), "results", "run_0")

        with (
            patch("nkululeko.utils.util.Util.get_path",
                  side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir_path + "/"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plots.plot_distributions(df)

        files = os.listdir(res_dir_path)
        assert any("value_counts_emotion_age" in f for f in files), f"Expected result file, got: {files}"

    def test_result_file_contains_overall_and_pairwise(self, tmp_path):
        """Result file has parseable 'overall:' and 'pairwise:' lines."""
        plots, _ = self._make_plots(tmp_path)
        df = _make_df()
        glob_conf.config["EXPL"]["value_counts"] = "[['age']]"

        fig_dir = os.path.join(str(tmp_path), "images")
        res_dir_path = os.path.join(str(tmp_path), "results", "run_0")

        with (
            patch("nkululeko.utils.util.Util.get_path",
                  side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir_path + "/"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plots.plot_distributions(df)

        res_file = os.path.join(res_dir_path, "value_counts_emotion_age.txt")
        assert os.path.isfile(res_file)
        with open(res_file) as f:
            lines = f.read().splitlines()
        overall_lines = [l for l in lines if l.startswith("overall:")]
        pairwise_lines = [l for l in lines if l.startswith("pairwise:")]
        assert overall_lines, "Expected at least one 'overall:' line"
        assert pairwise_lines, "Expected at least one 'pairwise:' line"
        # Lines must be parseable as Python dicts
        parsed_overall = ast.literal_eval(overall_lines[0][len("overall: "):])
        parsed_pairwise = ast.literal_eval(pairwise_lines[0][len("pairwise: "):])
        assert isinstance(parsed_overall, dict)
        assert isinstance(parsed_pairwise, dict)

    def test_no_result_file_for_two_continuous(self, tmp_path):
        """No result file is created when both columns are continuous (no categorical grouping)."""
        plots, _ = self._make_plots(tmp_path)
        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({
            "class_label": rng.normal(size=n),
            "age": rng.normal(size=n),
        })
        glob_conf.config["EXPL"]["value_counts"] = "[['age']]"

        fig_dir = os.path.join(str(tmp_path), "images")
        res_dir_path = os.path.join(str(tmp_path), "results", "run_0")

        with (
            patch("nkululeko.utils.util.Util.get_path",
                  side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir_path + "/"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plots.plot_distributions(df)

        files = os.listdir(res_dir_path)
        assert not any("value_counts_" in f for f in files), f"Unexpected result file: {files}"

    def test_result_file_deduplication(self, tmp_path):
        """Running plot_distributions twice does not duplicate lines in result file."""
        plots, _ = self._make_plots(tmp_path)
        df = _make_df()
        glob_conf.config["EXPL"]["value_counts"] = "[['age']]"

        fig_dir = os.path.join(str(tmp_path), "images")
        res_dir_path = os.path.join(str(tmp_path), "results", "run_0")

        with (
            patch("nkululeko.utils.util.Util.get_path",
                  side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir_path + "/"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plots.plot_distributions(df)
            plots.plot_distributions(df)

        res_file = os.path.join(res_dir_path, "value_counts_emotion_age.txt")
        with open(res_file) as f:
            lines = [l for l in f.read().splitlines() if l.startswith("overall:")]
        assert len(lines) == 1, f"Expected 1 overall: line, got {len(lines)}"
