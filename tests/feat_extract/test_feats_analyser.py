"""Tests for nkululeko/feat_extract/feats_analyser.py — FeatureAnalyser.

GH #459 item 5(c): feature_distributions should also get a speaker-level
plot (mirroring value_counts' existing speaker-level precedent), gated the
same way Experiment.plot_distribution() gates plot_distributions_speaker()
on got_speaker.
"""

from unittest.mock import MagicMock

import pandas as pd

from nkululeko.feat_extract.feats_analyser import FeatureAnalyser


def _make_instance(got_speaker, df_labels):
    instance = FeatureAnalyser.__new__(FeatureAnalyser)
    instance.df_labels = df_labels
    instance.features = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]})
    instance.context = type("Ctx", (), {"got_speaker": got_speaker})()
    instance.plots = MagicMock()
    return instance


class TestPlotFeaturesDistribution:
    def test_calls_plot_feature_speaker_when_speaker_ids_available(self):
        df_labels = pd.DataFrame(
            {"class_label": ["a", "a", "b"], "speaker": ["s1", "s1", "s2"]}
        )
        instance = _make_instance(got_speaker=True, df_labels=df_labels)

        instance._plot_features_distribution(["feat1"], "all")

        instance.plots.plot_feature.assert_called_once_with(
            "all", "feat1", "class_label", df_labels, instance.features
        )
        instance.plots.plot_feature_speaker.assert_called_once_with(
            "all", "feat1", "class_label", df_labels, instance.features
        )

    def test_skips_speaker_plot_when_got_speaker_is_false(self):
        df_labels = pd.DataFrame(
            {"class_label": ["a", "a", "b"], "speaker": ["s1", "s1", "s2"]}
        )
        instance = _make_instance(got_speaker=False, df_labels=df_labels)

        instance._plot_features_distribution(["feat1"], "all")

        instance.plots.plot_feature.assert_called_once()
        instance.plots.plot_feature_speaker.assert_not_called()

    def test_skips_speaker_plot_when_no_speaker_column(self):
        df_labels = pd.DataFrame({"class_label": ["a", "a", "b"]})
        instance = _make_instance(got_speaker=True, df_labels=df_labels)

        instance._plot_features_distribution(["feat1"], "all")

        instance.plots.plot_feature.assert_called_once()
        instance.plots.plot_feature_speaker.assert_not_called()

    def test_handles_multiple_features(self):
        df_labels = pd.DataFrame(
            {"class_label": ["a", "a", "b"], "speaker": ["s1", "s1", "s2"]}
        )
        instance = _make_instance(got_speaker=True, df_labels=df_labels)

        instance._plot_features_distribution(["feat1", "feat2"], "all")

        assert instance.plots.plot_feature.call_count == 2
        assert instance.plots.plot_feature_speaker.call_count == 2
