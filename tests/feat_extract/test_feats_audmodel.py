"""Tests for nkululeko/feat_extract/feats_audmodel.py — AudmodelSet.

Regression for GH #459:

1. Changing FEATS.audmodel.embeddings_name after a prior run silently
   reused the previous output head's cached values, because neither the
   whole-dataset store filename (extract()) nor the per-segment cache
   path (extract_sample_df()) included it.
2. Feature columns lost their real labels (e.g. "sadness", "arousal")
   and came back as plain integers 0..n-1.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nkululeko.feat_extract.feats_audmodel import AudmodelSet


def _make_instance():
    instance = AudmodelSet.__new__(AudmodelSet)
    instance.util = MagicMock()
    instance.model_loaded = True
    instance.use_torch = False
    instance.hidden_layer = 0
    return instance


class TestExtractSampleWithLabels:
    def test_multi_dim_head_labels_are_prefixed(self):
        instance = _make_instance()
        instance.embeddings_name = "emotion.acoustic"
        result = pd.DataFrame([[0.1, 0.7, 0.2]], columns=["anger", "sadness", "neutral"])
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )

        features, labels = instance._extract_sample_with_labels("signal", 16000)

        assert labels == [
            "emotion.acoustic_anger",
            "emotion.acoustic_sadness",
            "emotion.acoustic_neutral",
        ]
        assert list(features) == [0.1, 0.7, 0.2]

    def test_one_dimensional_head_label_is_just_the_head_name(self):
        instance = _make_instance()
        instance.embeddings_name = "valence"
        result = pd.DataFrame([[0.42]], columns=["valence"])
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )

        features, labels = instance._extract_sample_with_labels("signal", 16000)

        assert labels == ["valence"]
        assert list(features) == [0.42]

    def test_extract_sample_returns_plain_array_not_a_tuple(self):
        """extract_sample() keeps the generic Featureset contract (a plain
        feature array) since nkululeko.predict's generic extractor dispatch
        calls it directly and runs arithmetic (mean/std) on the result."""
        instance = _make_instance()
        instance.embeddings_name = "valence"
        result = pd.DataFrame([[0.42]], columns=["valence"])
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )

        features = instance.extract_sample("signal", 16000)

        assert not isinstance(features, tuple)
        assert list(features) == [0.42]


class TestCacheKeyIncludesEmbeddingsName:
    def _config_val(self, overrides):
        def side_effect(section, key, default):
            return overrides.get(key, default)

        return side_effect

    def test_extract_storage_filename_includes_embeddings_name(self):
        instance = _make_instance()
        instance.name = "mydb_feats"
        instance.context = MagicMock()
        instance.data_df = pd.DataFrame()
        instance.util.get_path.return_value = "/store/"
        instance.util.config_val.side_effect = self._config_val(
            {"store_format": "pkl", "audmodel.embeddings_name": "valence"}
        )
        instance._needs_extraction = MagicMock(return_value=False)
        instance.util.get_store = MagicMock(return_value=pd.DataFrame())

        instance.extract()

        storage_arg = instance.util.get_store.call_args[0][0]
        assert "valence" in storage_arg

    def test_different_embeddings_name_gives_different_storage_filename(self):
        results = {}
        for embeddings_name in ("valence", "arousal"):
            instance = _make_instance()
            instance.name = "mydb_feats"
            instance.context = MagicMock()
            instance.data_df = pd.DataFrame()
            instance.util.get_path.return_value = "/store/"
            instance.util.config_val.side_effect = self._config_val(
                {"store_format": "pkl", "audmodel.embeddings_name": embeddings_name}
            )
            instance._needs_extraction = MagicMock(return_value=False)
            instance.util.get_store = MagicMock(return_value=pd.DataFrame())
            instance.extract()
            results[embeddings_name] = instance.util.get_store.call_args[0][0]

        assert results["valence"] != results["arousal"]

    def test_extract_sample_df_cache_dir_includes_embeddings_name(self, tmp_path):
        instance = _make_instance()
        instance.embeddings_name = "emotion.acoustic"
        result = pd.DataFrame([[0.1, 0.9]], columns=["a", "b"])
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )
        instance.util.get_path.return_value = str(tmp_path)
        instance.util.config_val.side_effect = self._config_val(
            {"audmodel.id": "modelid", "audmodel.embeddings_name": "emotion.acoustic"}
        )

        index_tuple = ("f1.wav", pd.Timedelta(0), pd.Timedelta(seconds=1))
        df_part = instance.extract_sample_df(None, 16000, index_tuple)

        cache_dir = tmp_path / "modelid" / "emotion.acoustic"
        assert cache_dir.is_dir()
        assert list(df_part.columns) == ["emotion.acoustic_a", "emotion.acoustic_b"]

    def test_different_embeddings_name_gives_different_cache_dir(self, tmp_path):
        paths = {}
        for embeddings_name in ("valence", "arousal"):
            instance = _make_instance()
            instance.embeddings_name = embeddings_name
            result = pd.DataFrame([[0.5]], columns=[embeddings_name])
            instance.model_interface = MagicMock(
                process_signal=MagicMock(return_value=result)
            )
            instance.util.get_path.return_value = str(tmp_path)
            instance.util.config_val.side_effect = self._config_val(
                {"audmodel.id": "modelid", "audmodel.embeddings_name": embeddings_name}
            )
            index_tuple = ("f1.wav", pd.Timedelta(0), pd.Timedelta(seconds=1))
            instance.extract_sample_df(None, 16000, index_tuple)
            paths[embeddings_name] = tmp_path / "modelid" / embeddings_name

        assert paths["valence"] != paths["arousal"]
        assert paths["valence"].is_dir()
        assert paths["arousal"].is_dir()
