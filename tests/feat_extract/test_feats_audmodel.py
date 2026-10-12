"""Tests for nkululeko/feat_extract/feats_audmodel.py — AudmodelSet.

Regression for GH #459:

1. Changing FEATS.audmodel.embeddings_name after a prior run silently
   reused the previous output head's cached values, because neither the
   whole-dataset store filename (extract()) nor the per-segment cache
   path (extract_sample_df()) included it.
2. Feature columns lost their real labels (e.g. "sadness", "arousal")
   and came back as plain integers 0..n-1.
"""

from unittest.mock import MagicMock, patch

import audformat
import numpy as np
import pandas as pd
import pytest

from nkululeko.feat_extract.feats_audmodel import AudmodelSet, parse_embeddings_names


def _make_instance(embeddings_names=("valence",), head_label_counts=(1,)):
    instance = AudmodelSet.__new__(AudmodelSet)
    instance.util = MagicMock()
    instance.model_loaded = True
    instance.use_torch = False
    instance.hidden_layer = 0
    instance.embeddings_names = list(embeddings_names)
    instance.embeddings_name = instance.embeddings_names[0]
    instance._head_label_counts = list(head_label_counts)
    return instance


class TestExtractSampleWithLabels:
    def test_multi_dim_head_labels_are_prefixed(self):
        instance = _make_instance(["emotion.acoustic"], [3])
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
        instance = _make_instance(["valence"], [1])
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
        instance = _make_instance(["valence"], [1])
        result = pd.DataFrame([[0.42]], columns=["valence"])
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )

        features = instance.extract_sample("signal", 16000)

        assert not isinstance(features, tuple)
        assert list(features) == [0.42]

    def test_multi_head_labels_are_prefixed_per_head_boundary(self):
        """GH #459 item 3: with two configured heads extracted in one pass
        (concat=True), result.columns is both heads' labels concatenated
        with no boundary marker -- _head_label_counts (recorded at load
        time) must be used to prefix each column with its own head, not
        the first/only head."""
        instance = _make_instance(["valence", "emotion.acoustic"], [1, 3])
        result = pd.DataFrame(
            [[0.42, 0.1, 0.7, 0.2]],
            columns=["valence", "anger", "sadness", "neutral"],
        )
        instance.model_interface = MagicMock(
            process_signal=MagicMock(return_value=result)
        )

        features, labels = instance._extract_sample_with_labels("signal", 16000)

        assert labels == [
            "valence",
            "emotion.acoustic_anger",
            "emotion.acoustic_sadness",
            "emotion.acoustic_neutral",
        ]
        assert list(features) == [0.42, 0.1, 0.7, 0.2]


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

    def test_extract_storage_filename_includes_model_id(self):
        """Review on #459: self.name alone (built in feature_extractor.py
        from the database name + feats_type) doesn't vary with which
        audmodel.id is configured, so without the model id here too,
        switching models while keeping the same embeddings_name would
        silently reuse the previous model's cached whole-dataset features."""
        instance = _make_instance()
        instance.name = "mydb_feats"
        instance.context = MagicMock()
        instance.data_df = pd.DataFrame()
        instance.util.get_path.return_value = "/store/"
        instance.util.config_val.side_effect = self._config_val(
            {
                "store_format": "pkl",
                "audmodel.id": "1c250fd7-1.1.2",
                "audmodel.embeddings_name": "valence",
            }
        )
        instance._needs_extraction = MagicMock(return_value=False)
        instance.util.get_store = MagicMock(return_value=pd.DataFrame())

        instance.extract()

        storage_arg = instance.util.get_store.call_args[0][0]
        assert "1c250fd7-1.1.2" in storage_arg

    def test_different_model_id_gives_different_storage_filename(self):
        results = {}
        for model_id in ("model-a", "model-b"):
            instance = _make_instance()
            instance.name = "mydb_feats"
            instance.context = MagicMock()
            instance.data_df = pd.DataFrame()
            instance.util.get_path.return_value = "/store/"
            instance.util.config_val.side_effect = self._config_val(
                {
                    "store_format": "pkl",
                    "audmodel.id": model_id,
                    "audmodel.embeddings_name": "valence",
                }
            )
            instance._needs_extraction = MagicMock(return_value=False)
            instance.util.get_store = MagicMock(return_value=pd.DataFrame())
            instance.extract()
            results[model_id] = instance.util.get_store.call_args[0][0]

        assert results["model-a"] != results["model-b"]

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
        instance = _make_instance(["emotion.acoustic"], [2])
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


class TestParseEmbeddingsNames:
    """GH #459 item 3: FEATS.audmodel.embeddings_name must still accept a
    single bare name (back-compat) as well as a list literal for
    multi-head extraction."""

    def test_bare_word_stays_a_single_item_list(self):
        assert parse_embeddings_names("valence") == ["valence"]

    def test_dotted_name_stays_a_single_item_list(self):
        # "emotion.acoustic" isn't valid Python literal syntax either
        # (ast.literal_eval raises), same fallback path as a plain word.
        assert parse_embeddings_names("emotion.acoustic") == ["emotion.acoustic"]

    def test_list_literal_is_parsed(self):
        assert parse_embeddings_names("['arousal', 'valence']") == [
            "arousal",
            "valence",
        ]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            parse_embeddings_names("[]")


class _FakeOnnxModel:
    """Stand-in for audonnx.Model: only the bits _load_onnx_model() touches
    (.sess.get_providers(), .labels(), .outputs)."""

    def __init__(self, head_labels, providers=("CPUExecutionProvider",)):
        self._head_labels = head_labels
        self.outputs = dict(head_labels)
        self.sess = MagicMock(get_providers=MagicMock(return_value=list(providers)))

    def labels(self, outputs):
        if isinstance(outputs, (list, tuple)):
            result = []
            for name in outputs:
                result.extend(self._head_labels[name])
            return result
        return self._head_labels[outputs]


class TestLoadOnnxModelMultiHead:
    def _load(self, embeddings_names, fake_model, device="cpu"):
        instance = AudmodelSet.__new__(AudmodelSet)
        instance.util = MagicMock()
        instance.embeddings_names = embeddings_names
        instance.n_jobs = 1
        instance.device = device
        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audonnx.load",
                return_value=fake_model,
            ),
            patch(
                "nkululeko.feat_extract.feats_audmodel.audinterface.Feature"
            ) as feature_cls,
        ):
            instance._load_onnx_model("/fake/model/root")
        return instance, feature_cls

    def test_single_head_passes_plain_string_outputs_no_concat(self):
        fake_model = _FakeOnnxModel({"valence": ["valence"]})
        instance, feature_cls = self._load(["valence"], fake_model)

        assert instance._head_label_counts == [1]
        call_args, call_kwargs = feature_cls.call_args
        assert call_args[0] == ["valence"]
        assert call_kwargs["process_func_args"] == {"outputs": "valence"}

    def test_multi_head_passes_list_outputs_with_concat(self):
        fake_model = _FakeOnnxModel(
            {
                "valence": ["valence"],
                "emotion.acoustic": ["anger", "sadness", "neutral"],
            }
        )
        instance, feature_cls = self._load(["valence", "emotion.acoustic"], fake_model)

        assert instance._head_label_counts == [1, 3]
        call_args, call_kwargs = feature_cls.call_args
        assert call_args[0] == ["valence", "anger", "sadness", "neutral"]
        assert call_kwargs["process_func_args"] == {
            "outputs": ["valence", "emotion.acoustic"],
            "concat": True,
        }


class TestUnknownHeadValidation:
    """A typo'd/unknown FEATS.audmodel.embeddings_name used to surface as a
    bare `KeyError` from inside audonnx.Model.labels(), with no indication
    of what the valid output names actually are."""

    def _load(self, embeddings_names, fake_model):
        instance = AudmodelSet.__new__(AudmodelSet)
        instance.util = MagicMock()
        instance.util.error.side_effect = RuntimeError("boom")
        instance.embeddings_names = embeddings_names
        instance.n_jobs = 1
        instance.device = "cpu"
        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audonnx.load",
                return_value=fake_model,
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audinterface.Feature"),
            pytest.raises(RuntimeError),
        ):
            instance._load_onnx_model("/fake/model/root")
        return instance

    def test_unknown_head_lists_valid_names_instead_of_keyerror(self):
        fake_model = _FakeOnnxModel(
            {"valence": ["valence"], "arousal": ["arousal"]}
        )

        instance = self._load(["valennce"], fake_model)

        message = instance.util.error.call_args[0][0]
        assert "valennce" in message
        assert "arousal" in message and "valence" in message

    def test_one_valid_one_unknown_head_still_reports_the_bad_one(self):
        fake_model = _FakeOnnxModel(
            {"valence": ["valence"], "arousal": ["arousal"]}
        )

        instance = self._load(["valence", "arousall"], fake_model)

        message = instance.util.error.call_args[0][0]
        assert "arousall" in message
        assert "valence" not in message.split(";")[0]


class TestGpuFallbackWarning:
    """GH #459 item 5(a): onnxruntime silently falls back to CPU when CUDA
    was requested but isn't actually available (e.g. missing cuDNN 9)."""

    def _load(self, providers, device):
        fake_model = _FakeOnnxModel({"valence": ["valence"]}, providers=providers)
        instance = AudmodelSet.__new__(AudmodelSet)
        instance.util = MagicMock()
        instance.embeddings_names = ["valence"]
        instance.n_jobs = 1
        instance.device = device
        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audonnx.load",
                return_value=fake_model,
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audinterface.Feature"),
        ):
            instance._load_onnx_model("/fake/model/root")
        return instance

    def test_warns_when_cuda_requested_but_fell_back_to_cpu(self):
        instance = self._load(("CPUExecutionProvider",), "cuda")

        assert instance.util.warn.called
        message = instance.util.warn.call_args[0][0]
        assert "CUDA" in message

    def test_no_warning_when_cuda_is_actually_active(self):
        instance = self._load(
            ("CUDAExecutionProvider", "CPUExecutionProvider"), "cuda"
        )

        assert not instance.util.warn.called

    def test_no_warning_when_cpu_was_requested(self):
        instance = self._load(("CPUExecutionProvider",), "cpu")

        assert not instance.util.warn.called


class TestMultiHeadRejectedForTorchPath:
    def test_multi_head_with_nonzero_hidden_layer_errors(self):
        instance = AudmodelSet.__new__(AudmodelSet)
        instance.util = MagicMock()
        instance.util.error.side_effect = RuntimeError("boom")
        instance.util.config_val.side_effect = lambda section, key, default: {
            ("FEATS", "audmodel.id"): "modelid",
            ("FEATS", "audmodel.embeddings_name"): "['arousal', 'valence']",
            ("MODEL", "device"): "cpu",
        }.get((section, key), default)
        instance.hidden_layer = 1

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audmodel.load",
                return_value="/fake/root",
            ),
            patch(
                "nkululeko.feat_extract.feats_audmodel.os.path.isfile",
                return_value=False,
            ),
            pytest.raises(RuntimeError),
        ):
            instance._load_model()

        assert instance.util.error.called


class TestExtractPreservesFailedRows:
    """Review on #468: a single bad file used to abort the whole
    extraction (self.util.error() raises), which also meant
    ap_audmodel.py's AudmodelPredictor could never actually reach its
    all-zero-row "unknown" fallback -- that path was dead code. A failed
    file now becomes a NaN row (once concatenated with the columns other
    rows have) instead of aborting."""

    def test_failed_file_becomes_nan_row_instead_of_aborting(self):
        instance = _make_instance()
        instance.name = "mydb_feats"
        instance.context = MagicMock()
        instance.data_df = pd.DataFrame(
            index=audformat.segmented_index(["a.wav", "b.wav"], [0, 0], [1, 1])
        )
        instance.util.get_path.return_value = "/store/"
        instance.util.config_val.side_effect = lambda section, key, default: default
        instance._needs_extraction = MagicMock(return_value=True)
        instance.util.write_store = MagicMock()

        good_df = pd.DataFrame(
            [[0.5]],
            columns=["valence"],
            index=audformat.segmented_index(["a.wav"], [0], [1]),
        )

        def fake_extract_sample_df(signal, sr, index_tuple):
            if index_tuple[0] == "b.wav":
                raise RuntimeError("boom")
            return good_df

        instance.extract_sample_df = MagicMock(side_effect=fake_extract_sample_df)

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audiofile.read",
                return_value=(np.array([0.0]), 16000),
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audiofile.write"),
        ):
            instance.extract()

        assert not instance.util.error.called
        assert instance.util.warn.called
        assert len(instance.df) == 2
        a_key = ("a.wav", pd.Timedelta(0), pd.Timedelta(seconds=1))
        b_key = ("b.wav", pd.Timedelta(0), pd.Timedelta(seconds=1))
        assert instance.df.loc[a_key, "valence"] == 0.5
        assert pd.isna(instance.df.loc[b_key, "valence"])

    def test_every_file_failing_leaves_an_all_nan_frame(self):
        """If every row fails but FEATS.fail_threshold tolerates it (set to
        1.0 here to isolate this from the threshold-enforcement behavior,
        covered separately in TestFailThresholdEnforcement), the resulting
        store has no columns at all (matching ap_audmodel.py's "no
        head_cols found at all" branch)."""
        instance = _make_instance()
        instance.name = "mydb_feats"
        instance.context = MagicMock()
        instance.data_df = pd.DataFrame(
            index=audformat.segmented_index(["a.wav"], [0], [1])
        )
        instance.util.get_path.return_value = "/store/"
        instance.util.config_val.side_effect = (
            lambda section, key, default: "1.0"
            if key == "fail_threshold"
            else default
        )
        instance._needs_extraction = MagicMock(return_value=True)
        instance.util.write_store = MagicMock()
        instance.extract_sample_df = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audiofile.read",
                return_value=(np.array([0.0]), 16000),
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audiofile.write"),
        ):
            instance.extract()

        assert not instance.util.error.called
        assert instance.df.shape[1] == 0
        assert len(instance.df) == 1


class TestFailThresholdEnforcement:
    """Review on #468: preserving a failed file as a NaN row (instead of
    aborting) must not bypass FEATS.fail_threshold -- a store that's
    entirely or mostly empty must not be persisted and later reused as if
    extraction had succeeded."""

    def _instance(self, fail_threshold=None):
        instance = _make_instance()
        instance.name = "mydb_feats"
        instance.context = MagicMock()
        instance.util.get_path.return_value = "/store/"
        overrides = {}
        if fail_threshold is not None:
            overrides["fail_threshold"] = str(fail_threshold)
        instance.util.config_val.side_effect = (
            lambda section, key, default: overrides.get(key, default)
        )
        instance._needs_extraction = MagicMock(return_value=True)
        instance.util.write_store = MagicMock()
        return instance

    def test_failure_rate_exceeding_default_threshold_raises(self):
        instance = self._instance()
        instance.util.error.side_effect = RuntimeError("boom")
        instance.data_df = pd.DataFrame(
            index=audformat.segmented_index(["a.wav", "b.wav"], [0, 0], [1, 1])
        )
        instance.extract_sample_df = MagicMock(side_effect=RuntimeError("extraction failed"))

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audiofile.read",
                return_value=(np.array([0.0]), 16000),
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audiofile.write"),
            pytest.raises(RuntimeError),
        ):
            instance.extract()

        assert instance.util.error.called
        assert not instance.util.write_store.called

    def test_failure_rate_within_default_threshold_does_not_raise(self):
        instance = self._instance()
        instance.data_df = pd.DataFrame(
            index=audformat.segmented_index(
                ["a.wav", "b.wav", "c.wav", "d.wav"], [0, 0, 0, 0], [1, 1, 1, 1]
            )
        )
        good_df = pd.DataFrame(
            [[0.5]],
            columns=["valence"],
            index=audformat.segmented_index(["ok"], [0], [1]),
        )

        def fake_extract_sample_df(signal, sr, index_tuple):
            if index_tuple[0] == "a.wav":
                raise RuntimeError("boom")
            return good_df

        instance.extract_sample_df = MagicMock(side_effect=fake_extract_sample_df)

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audiofile.read",
                return_value=(np.array([0.0]), 16000),
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audiofile.write"),
        ):
            instance.extract()

        assert not instance.util.error.called
        assert instance.util.write_store.called
        assert len(instance.df) == 4

    def test_custom_fail_threshold_is_honored(self):
        """A 67% failure rate exceeds the 0.5 default but is within a
        custom, higher threshold -- must not raise."""
        instance = self._instance(fail_threshold=0.9)
        instance.data_df = pd.DataFrame(
            index=audformat.segmented_index(
                ["a.wav", "b.wav", "c.wav"], [0, 0, 0], [1, 1, 1]
            )
        )
        good_df = pd.DataFrame(
            [[0.5]],
            columns=["valence"],
            index=audformat.segmented_index(["ok"], [0], [1]),
        )

        def fake_extract_sample_df(signal, sr, index_tuple):
            if index_tuple[0] in ("a.wav", "b.wav"):
                raise RuntimeError("boom")
            return good_df

        instance.extract_sample_df = MagicMock(side_effect=fake_extract_sample_df)

        with (
            patch(
                "nkululeko.feat_extract.feats_audmodel.audiofile.read",
                return_value=(np.array([0.0]), 16000),
            ),
            patch("nkululeko.feat_extract.feats_audmodel.audiofile.write"),
        ):
            instance.extract()

        assert not instance.util.error.called
        assert instance.util.write_store.called
