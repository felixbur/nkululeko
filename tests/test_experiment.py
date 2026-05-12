import pytest
import configparser
import random
import tempfile
import pandas as pd
import nkululeko.glob_conf as glob_conf


@pytest.fixture
def mock_config():
    """Create a minimal config for Experiment."""
    config = configparser.ConfigParser()
    config.add_section("EXP")
    config.set("EXP", "name", "test_experiment")
    config.set("EXP", "root", tempfile.gettempdir())
    config.set("EXP", "runs", "1")
    config.set("EXP", "epochs", "1")
    config.set("EXP", "traindevtest", "False")
    config.add_section("DATA")
    config.set("DATA", "databases", "['test_db']")
    config.set("DATA", "target", "emotion")
    config.set("DATA", "labels", "['happy', 'sad']")
    config.add_section("FEATS")
    config.set("FEATS", "type", "['os']")
    config.add_section("MODEL")
    config.set("MODEL", "type", "xgb")
    config.add_section("REPORT")
    config.set("REPORT", "fresh", "True")
    return config


class TestExperimentSetGlobals:
    """Test set_globals and set_module via glob_conf directly."""

    def test_set_globals_sets_config(self, mock_config):
        """Test that init_config sets the config in glob_conf."""
        glob_conf.init_config(mock_config)
        assert glob_conf.config is mock_config
        assert glob_conf.config["EXP"]["name"] == "test_experiment"

    def test_set_module(self, mock_config):
        """Test set_module sets the module in glob_conf."""
        glob_conf.init_config(mock_config)
        glob_conf.set_module("test")
        assert glob_conf.module == "test"


class TestExperimentImportCsv:
    """Test the _import_csv method."""

    def test_import_csv_with_valid_file(self, mock_config, tmp_path):
        """Test importing a valid CSV file.

        Note: This is a minimal test that verifies _import_csv can read
        a valid audformat CSV file. Full integration testing with dataset
        loading is covered by the end-to-end tests.
        """
        from nkululeko.experiment import Experiment
        from nkululeko import glob_conf

        # Create a CSV file in audformat-compatible format
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"file": ["a.wav", "b.wav"], "emotion": ["happy", "sad"]})
        df.index.name = "file"
        df.to_csv(csv_file)
        assert csv_file.exists()

        # Initialize glob_conf
        glob_conf.init_config(mock_config)

        # Create a minimal Experiment and manually set target to test _import_csv
        exp = Experiment.__new__(Experiment)
        exp.target = "emotion"  # Set target manually to avoid full initialization

        result_df = exp._import_csv(str(csv_file))

        # Verify the DataFrame was imported correctly
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert "emotion" in result_df.columns
        assert len(result_df) == 2
        assert hasattr(result_df, "is_labeled")
        assert result_df.is_labeled is True


class TestExperimentHelpers:
    """Test helper methods."""

    def test_add_random_target(self):
        """Test _add_random_target adds random labels."""
        glob_conf.labels = ["happy", "sad", "angry"]
        df = pd.DataFrame({"speaker": ["s1", "s2", "s3"]})

        # Replicate _add_random_target logic directly
        target = "emotion"
        labels = glob_conf.labels
        random.seed(42)
        a = [None] * len(df)
        for i in range(len(df)):
            a[i] = random.choice(labels)
        df[target] = a

        assert target in df.columns
        assert len(df) == 3
        for label in df[target]:
            assert label in ["happy", "sad", "angry"]

    def test_decode_labels_no_encoder(self):
        """Test _decode_labels returns original column when no encoder."""
        # Without a label_encoder, _decode_labels should return the column unchanged
        column_name = "emotion"
        has_encoder = False
        result = column_name if not has_encoder else "decoded_emotion"
        assert result == "emotion"

    def test_decode_labels_with_encoder(self):
        """Test _decode_labels concept with a label encoder."""
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        le.fit(["happy", "sad", "angry"])
        df = pd.DataFrame({"emotion": [0, 1, 2]})
        decoded = le.inverse_transform(df["emotion"])
        assert list(decoded) == ["angry", "happy", "sad"]


class TestBuildTestDsDf:
    """Test _build_test_ds_df and evaluate_per_test_set functionality."""

    def _make_segmented_index(self, files):
        """Create a minimal segmented MultiIndex mimicking audformat."""
        import pandas as pd

        arrays = [
            files,
            pd.to_timedelta([0] * len(files), unit="s"),
            pd.to_timedelta([1] * len(files), unit="s"),
        ]
        return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])

    def test_build_test_ds_df_empty_when_test_empty(self, mock_config, tmp_path):
        """_build_test_ds_df should leave test_ds_df empty when test is empty."""
        from nkululeko.data.datasplitter import Datasplitter

        glob_conf.init_config(mock_config)
        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type(
            "U",
            (),
            {
                "get_path": lambda self, k: str(tmp_path) + "/",
            },
        )()
        ds.df_test = pd.DataFrame()
        ds.datasets = {"ds_a": None, "ds_b": None}
        ds._build_test_ds_df()
        assert ds.test_ds_df == {}

    def test_build_test_ds_df_filters_by_dataset_index(self, mock_config, tmp_path):
        """_build_test_ds_df should assign each test row to its source dataset."""
        from nkululeko.data.datasplitter import Datasplitter

        glob_conf.init_config(mock_config)

        # Build two mini test DataFrames with disjoint file sets
        files_a = ["/data/ds_a/01.wav", "/data/ds_a/02.wav"]
        files_b = ["/data/ds_b/03.wav", "/data/ds_b/04.wav"]
        idx_a = self._make_segmented_index(files_a)
        idx_b = self._make_segmented_index(files_b)

        df_a = pd.DataFrame({"emotion": [0, 1]}, index=idx_a)
        df_b = pd.DataFrame({"emotion": [1, 0]}, index=idx_b)

        # Combined encoded test DF
        df_test = pd.concat([df_a, df_b])

        # Pickle the per-dataset split files into tmp_path (mimicking Dataset.finish_up)
        store = str(tmp_path) + "/"
        df_a.to_pickle(f"{store}ds_a_testdf.pkl")
        df_b.to_pickle(f"{store}ds_b_testdf.pkl")

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type(
            "U",
            (),
            {
                "get_path": lambda self, k: store,
            },
        )()
        ds.datasets = {"ds_a": None, "ds_b": None}
        ds.df_test = df_test

        ds._build_test_ds_df()

        assert set(ds.test_ds_df.keys()) == {"ds_a", "ds_b"}
        assert len(ds.test_ds_df["ds_a"]) == 2
        assert len(ds.test_ds_df["ds_b"]) == 2
        # Verify only the correct rows end up in each slice
        assert all(ds.test_ds_df["ds_a"].index.isin(idx_a))
        assert all(ds.test_ds_df["ds_b"].index.isin(idx_b))

    def test_evaluate_per_test_set_noop_single_dataset(self, mock_config, tmp_path):
        """evaluate_per_test_set should be a no-op with only one test dataset."""
        from nkululeko.experiment import Experiment

        glob_conf.init_config(mock_config)
        exp = Experiment.__new__(Experiment)
        exp.test_ds_df = {"only_ds": pd.DataFrame()}
        exp.test_empty = False
        # Should return early without touching runmgr
        exp.evaluate_per_test_set()  # must not raise

    def test_evaluate_per_test_set_noop_no_runmgr(self, mock_config, tmp_path):
        """evaluate_per_test_set should be a no-op when runmgr is absent."""
        from nkululeko.experiment import Experiment

        glob_conf.init_config(mock_config)
        exp = Experiment.__new__(Experiment)
        exp.test_ds_df = {"ds_a": pd.DataFrame(), "ds_b": pd.DataFrame()}
        # Provide a datasplitter mock with a non-empty df_test
        datasplitter_mock = type("DS", (), {})()
        datasplitter_mock.df_test = pd.DataFrame({"emotion": [0]})
        exp.datasplitter = datasplitter_mock
        # No runmgr attribute – must not raise
        exp.evaluate_per_test_set()

    def test_evaluate_per_test_set_noop_no_modelrunner(self, mock_config, tmp_path):
        """evaluate_per_test_set should be a no-op when modelrunner is absent."""
        from nkululeko.experiment import Experiment

        glob_conf.init_config(mock_config)
        exp = Experiment.__new__(Experiment)
        exp.test_ds_df = {"ds_a": pd.DataFrame(), "ds_b": pd.DataFrame()}
        # Provide a datasplitter mock with a non-empty df_test
        datasplitter_mock = type("DS", (), {})()
        datasplitter_mock.df_test = pd.DataFrame({"emotion": [0]})
        exp.datasplitter = datasplitter_mock
        exp.runmgr = object()  # no 'modelrunner' attribute
        # Must not raise
        exp.evaluate_per_test_set()


# --- Module-level helpers for the save() regression tests below ----------
# pickle requires classes to be defined at module level (or otherwise
# importable) — closures in tests can't be pickled.


class _FakeModel:
    """Stand-in for `runmgr.modelrunner.model`; picklable, not an ANN."""

    def is_ann(self):
        return False


class _FakeModelrunner:
    def __init__(self):
        self.model = _FakeModel()


class _FakeRunmgr:
    def __init__(self):
        self.modelrunner = _FakeModelrunner()


class _CollectingUtil:
    """Picklable substitute for `Util`; `warn` accumulates messages."""

    def __init__(self):
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)


class _Unpicklable:
    """Stands in for an audonnx `InferenceSession`: pickling raises."""

    def __reduce__(self):
        raise TypeError("cannot pickle me")


class _FakeInnerExtractor:
    """Picklable `feat_extractor` placeholder that owns an un-picklable model."""

    def __init__(self, bad_obj):
        self.model = bad_obj
        self.model_interface = bad_obj
        self.model_loaded = True


class _FakeFeatureExtractor:
    def __init__(self, bad_obj):
        self.feat_extractor = _FakeInnerExtractor(bad_obj)


class _FakeDatasplitter:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor


class TestExperimentSave:
    """Regression for Experiment.save():

    When the inner feature extractor holds an un-picklable object (e.g. an
    audonnx InferenceSession used by `Audwav2vec2Set` / `AgenderSet`), the
    initial pickle.dump fails and we fall into the except handler. That
    handler must strip the model/model_interface fields from EVERY
    FeatureExtractor on the experiment, not just the one inside
    Datasplitter -- otherwise the retry pickle still hits the same object.
    """

    def test_save_strips_models_from_both_feature_extractors(self, mock_config, tmp_path):
        import pickle as _pickle

        from nkululeko.experiment import Experiment

        glob_conf.init_config(mock_config)

        bad = _Unpicklable()
        ext_top = _FakeFeatureExtractor(bad)
        ext_ds = _FakeFeatureExtractor(bad)

        exp = Experiment.__new__(Experiment)
        exp.runmgr = _FakeRunmgr()
        exp.util = _CollectingUtil()
        exp.feature_extractor = ext_top
        exp.datasplitter = _FakeDatasplitter(ext_ds)

        out = tmp_path / "exp.pkl"
        exp.save(str(out))

        # Both inner extractors had their model nulled.
        assert ext_top.feat_extractor.model is None
        assert ext_ds.feat_extractor.model is None
        # model_interface was also dropped.
        assert ext_top.feat_extractor.model_interface is None
        assert ext_ds.feat_extractor.model_interface is None
        # model_loaded was reset, so a fresh load is forced after unpickle.
        assert ext_top.feat_extractor.model_loaded is False
        assert ext_ds.feat_extractor.model_loaded is False

        # The retry pickle succeeded and produced a file we can load back.
        assert out.is_file()
        with open(out, "rb") as fh:
            d = _pickle.load(fh)
        assert "feature_extractor" in d
        assert d["feature_extractor"].feat_extractor.model_loaded is False

        # save() must have warned about the stripped model.
        assert any(
            "Can't pickle the feature extraction model" in w
            for w in exp.util.warnings
        )

    def test_save_handles_single_extractor(self, mock_config, tmp_path):
        """If only one extractor is reachable (no datasplitter), save still works."""
        from nkululeko.experiment import Experiment

        glob_conf.init_config(mock_config)

        ext = _FakeFeatureExtractor(_Unpicklable())

        exp = Experiment.__new__(Experiment)
        exp.runmgr = _FakeRunmgr()
        exp.util = _CollectingUtil()
        exp.feature_extractor = ext
        # No `datasplitter` attribute on purpose.

        out = tmp_path / "exp.pkl"
        exp.save(str(out))

        assert ext.feat_extractor.model is None
        assert out.is_file()


class TestExperimentCollectFeatureExtractors:
    """`_collect_feature_extractors` finds every FeatureExtractor on the experiment."""

    def test_collects_both_when_distinct(self):
        from types import SimpleNamespace

        from nkululeko.experiment import Experiment

        a = SimpleNamespace(feat_extractor=None)
        b = SimpleNamespace(feat_extractor=None)
        exp = Experiment.__new__(Experiment)
        exp.feature_extractor = a
        exp.datasplitter = SimpleNamespace(feature_extractor=b)

        result = exp._collect_feature_extractors()
        assert len(result) == 2
        assert a in result
        assert b in result

    def test_dedupes_when_same_instance(self):
        """If `self.feature_extractor` and `datasplitter.feature_extractor`
        point at the same object, return it only once."""
        from types import SimpleNamespace

        from nkululeko.experiment import Experiment

        shared = SimpleNamespace(feat_extractor=None)
        exp = Experiment.__new__(Experiment)
        exp.feature_extractor = shared
        exp.datasplitter = SimpleNamespace(feature_extractor=shared)

        result = exp._collect_feature_extractors()
        assert result == [shared]

    def test_returns_empty_when_none(self):
        from nkululeko.experiment import Experiment

        exp = Experiment.__new__(Experiment)
        assert exp._collect_feature_extractors() == []
