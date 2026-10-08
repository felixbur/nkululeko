"""Tests for nkululeko/plots.py — Plots.plot_distributions_speaker().

Regression for GH #462: plot_distributions_speaker() represented each
speaker by its first sample (df_speaker.head(1)), so for a within-speaker
design (every speaker has samples of every target class, e.g. paired
sober/intoxicated recordings of the same speakers), every speaker was
counted only for the class of its first sample -- wrong speaker-level
distributions, and a crash downstream (find_most_significant_difference
needs >= 2 groups) once the speaker-level data collapsed to a single class.
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf


class TestDedupeSpeakersForDistribution:
    """Unit-level coverage of the extracted dedup helper, no plotting
    involved."""

    def _make_plots(self):
        from nkululeko.plots import Plots

        with patch("nkululeko.utils.util.Util.get_path", return_value="/tmp/"):
            return Plots()

    def test_within_speaker_design_keeps_one_row_per_speaker_per_class(self):
        """The core regression: every speaker has samples of every class."""
        df = pd.DataFrame(
            {
                "speaker": ["s1", "s1", "s2", "s2"],
                "class_label": ["sober", "intoxicated", "sober", "intoxicated"],
            }
        )
        plots = self._make_plots()
        df_speakers = df.groupby("speaker").head(1)  # old, buggy dedup
        out = plots._dedupe_speakers_for_distribution(df, df_speakers)

        assert len(out) == 4  # 2 speakers x 2 classes each
        assert set(out["class_label"]) == {"sober", "intoxicated"}
        for speaker in ("s1", "s2"):
            classes = set(out.loc[out.speaker == speaker, "class_label"])
            assert classes == {"sober", "intoxicated"}

    def test_speaker_with_constant_class_still_counted_once(self):
        df = pd.DataFrame(
            {
                "speaker": ["s1", "s1", "s1", "s2"],
                "class_label": ["happy", "happy", "happy", "sad"],
            }
        )
        plots = self._make_plots()
        df_speakers = df.groupby("speaker").head(1)
        out = plots._dedupe_speakers_for_distribution(df, df_speakers)

        assert len(out) == 2
        assert set(out["class_label"]) == {"happy", "sad"}

    def test_no_class_label_column_falls_back_to_given_df_speakers(self):
        df = pd.DataFrame({"speaker": ["s1", "s1", "s2"]})
        plots = self._make_plots()
        df_speakers = df.groupby("speaker").head(1)
        out = plots._dedupe_speakers_for_distribution(df, df_speakers)

        assert out is df_speakers

    def test_regression_target_falls_back_to_given_df_speakers(self):
        """Review on #462: class_label is also present for regression
        targets (continuous values), not just classification. Grouping by
        its exact value would put almost every sample in its own group
        (since continuous measurements rarely repeat exactly), reverting
        these plots to sample-level weighting instead of fixing anything
        -- so the per-class expansion must only apply to categorical
        (classification) targets."""
        df = pd.DataFrame(
            {
                "speaker": ["s1", "s1", "s1", "s2", "s2"],
                "class_label": [23.1, 23.4, 22.9, 41.0, 40.5],
            }
        )
        plots = self._make_plots()
        df_speakers = df.groupby("speaker").head(1)
        out = plots._dedupe_speakers_for_distribution(df, df_speakers)

        assert out is df_speakers
        assert len(out) == 2


class TestFindMostSignificantDifferenceSafe:
    """GH #462: a statistic that ends up with fewer than 2 groups (e.g.
    after the within-speaker dedup above collapses to one group) must warn
    and skip instead of raising and aborting the whole explore run."""

    def _make_plots(self):
        from nkululeko.plots import Plots

        with patch("nkululeko.utils.util.Util.get_path", return_value="/tmp/"):
            plots = Plots()
        plots.util = MagicMock()
        return plots

    def test_single_group_warns_and_returns_none(self):
        plots = self._make_plots()
        pairwise, overall = plots._find_most_significant_difference_safe(
            {"only_group": [1, 2, 3]}, mean_featnum=10, context="test"
        )
        assert pairwise is None
        assert overall is None
        assert plots.util.warn.called

    def test_two_groups_computes_normally(self):
        plots = self._make_plots()
        pairwise, overall = plots._find_most_significant_difference_safe(
            {"a": [1, 2, 3], "b": [10, 11, 12]}, mean_featnum=10, context="test"
        )
        assert pairwise is not None
        assert not plots.util.warn.called


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    import configparser

    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "intoxication", "databases": "['data']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    # Numeric (continuous) value_counts attribute, matching the issue's own
    # repro (value_counts = [['gender'], ['age'], ['duration']]) -- a
    # categorical-vs-categorical attribute like gender never reaches
    # _save_distribution_stats at all, so it wouldn't reproduce the crash.
    config["EXPL"] = {"value_counts": "[['duration']]"}
    config["PLOT"] = {"format": "png", "titles": "False"}
    glob_conf.config = config
    glob_conf.report = MagicMock()
    yield
    glob_conf.config = None


class TestPlotDistributionsSpeakerWithinSpeakerDesign:
    """End-to-end: plot_distributions_speaker() must not crash for a
    within-speaker design, and must reflect both classes per speaker."""

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
        return plots, fig_dir, res_dir

    def test_does_not_crash_when_every_speaker_has_every_class(self, tmp_path):
        """GH #462's actual reported crash: with the pre-fix dedup, every
        speaker collapses to a single class (whichever sorts first), so
        the speaker-level stats call ends up with only one group and
        find_most_significant_difference raised "Need at least 2
        distributions for comparison", aborting the whole explore run."""
        plots, fig_dir, res_dir = self._make_plots(tmp_path)
        rng_duration = iter([1.0 + 0.01 * i for i in range(400)])
        rows = []
        for speaker in [f"spk{i}" for i in range(20)]:
            for cls in ("sober", "intoxicated"):
                for _ in range(10):
                    rows.append(
                        {
                            "speaker": speaker,
                            "class_label": cls,
                            "duration": next(rng_duration),
                        }
                    )
        df = pd.DataFrame(rows)

        with (
            patch(
                "nkululeko.utils.util.Util.get_path",
                side_effect=lambda p: fig_dir + "/" if p == "fig_dir" else res_dir + "/",
            ),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            plots.plot_distributions_speaker(df)  # must not raise

        res_file = os.path.join(res_dir, "value_counts_intoxication_duration.txt")
        assert os.path.isfile(res_file)
        with open(res_file) as f:
            content = f.read()
        assert "overall:" in content or "pairwise:" in content
