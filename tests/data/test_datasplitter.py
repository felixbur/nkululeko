"""Tests for nkululeko/data/datasplitter.py — Datasplitter class."""

import configparser
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.data.datasplitter import Datasplitter


def _make_segmented_index(files):
    arrays = [
        files,
        [timedelta(0)] * len(files),
        [timedelta(seconds=1)] * len(files),
    ]
    return pd.MultiIndex.from_arrays(arrays, names=["file", "start", "end"])


def _tag_df(df):
    """Set standard boolean flags to False on a split DataFrame."""
    df.is_labeled = False
    df.got_gender = False
    df.got_speaker = False
    return df


def _make_fake_util(tmp_path):
    """Return a minimal no-op utility stub for Datasplitter tests."""
    util = MagicMock()
    util.get_path.return_value = str(tmp_path) + "/"
    util.config_val.side_effect = lambda sec, key, default: default
    util.exp_is_classification.return_value = False
    return util


@pytest.fixture(autouse=True)
def setup_glob_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {
        "type": "classification",
        "name": "test_ds",
        "root": str(tmp_path),
        "runs": "1",
        "epochs": "1",
        "traindevtest": "False",
    }
    config["DATA"] = {
        "target": "emotion",
        "databases": "['test_db']",
        "labels": "['happy', 'sad', 'angry']",
    }
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    glob_conf.init_config(config)
    glob_conf.labels = ["happy", "sad", "angry"]
    glob_conf.target = "emotion"
    glob_conf.split3 = False
    yield
    glob_conf.config = None
    glob_conf.labels = None
    glob_conf.target = None


@pytest.fixture
def ds_bare(tmp_path):
    """A Datasplitter created with __new__ (no datasets needed)."""
    ds = Datasplitter.__new__(Datasplitter)
    ds.util = type(
        "U",
        (),
        {
            "get_path": lambda self, k: str(tmp_path) + "/",
            "warn": lambda self, m: None,
        },
    )()
    ds.datasets = {}
    ds.target = "emotion"
    ds.split3 = False
    ds.got_speaker = False
    return ds


class TestAddRandomTarget:
    def test_adds_target_column(self, ds_bare):
        df = pd.DataFrame({"speaker": ["s1", "s2", "s3"]})
        result = ds_bare._add_random_target(df)
        assert "emotion" in result.columns

    def test_all_labels_from_glob_labels(self, ds_bare):
        df = pd.DataFrame(index=range(100))
        result = ds_bare._add_random_target(df)
        assert set(result["emotion"]).issubset({"happy", "sad", "angry"})

    def test_returns_same_length(self, ds_bare):
        df = pd.DataFrame(index=range(10))
        result = ds_bare._add_random_target(df)
        assert len(result) == 10


class TestGetSampleSelection:
    def _make_ds_with_splits(self, tmp_path):
        from nkululeko.utils.util import Util

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = Util("datasplitter")
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {}
        idx_tr = _make_segmented_index(["/data/tr_1.wav", "/data/tr_2.wav"])
        idx_te = _make_segmented_index(["/data/te_1.wav"])
        ds.df_train = pd.DataFrame({"emotion": [0, 1]}, index=idx_tr)
        ds.df_test = pd.DataFrame({"emotion": [0]}, index=idx_te)
        return ds

    def test_all_returns_train_and_test(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "all"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 3

    def test_train_returns_only_train(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "train"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 2

    def test_test_returns_only_test(self, tmp_path):
        glob_conf.config["EXP"]["sample_selection"] = "test"
        ds = self._make_ds_with_splits(tmp_path)
        result = ds.get_sample_selection()
        assert len(result) == 1


class TestBuildTestDsDf:
    def test_empty_test_produces_empty_mapping(self, ds_bare):
        ds_bare.df_test = pd.DataFrame()
        ds_bare._build_test_ds_df()
        assert ds_bare.test_ds_df == {}

    def test_in_memory_split_used_preferentially(self, tmp_path):
        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False

        files_a = ["/data/a_1.wav", "/data/a_2.wav"]
        idx_a = _make_segmented_index(files_a)
        df_a = pd.DataFrame({"emotion": [0, 1]}, index=idx_a)

        mock_ds = type("DS", (), {"df_test": df_a})()
        ds.datasets = {"db_a": mock_ds}
        ds.df_test = df_a
        ds._build_test_ds_df()

        assert "db_a" in ds.test_ds_df
        assert len(ds.test_ds_df["db_a"]) == 2

    def test_falls_back_to_pkl_when_no_in_memory(self, tmp_path):
        files = ["/data/f_1.wav", "/data/f_2.wav"]
        idx = _make_segmented_index(files)
        df = pd.DataFrame({"emotion": [0, 1]}, index=idx)
        df.to_pickle(str(tmp_path) + "/mydb_testdf.pkl")

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"mydb": None}  # None → no in-memory split
        ds.df_test = df
        ds._build_test_ds_df()

        assert "mydb" in ds.test_ds_df

    def test_two_datasets_correctly_separated(self, tmp_path):
        files_a = ["/data/a.wav"]
        files_b = ["/data/b.wav"]
        idx_a = _make_segmented_index(files_a)
        idx_b = _make_segmented_index(files_b)
        df_a = pd.DataFrame({"emotion": [0]}, index=idx_a)
        df_b = pd.DataFrame({"emotion": [1]}, index=idx_b)
        df_test = pd.concat([df_a, df_b])

        ds_a = type("DS", (), {"df_test": df_a})()
        ds_b = type("DS", (), {"df_test": df_b})()

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = type("U", (), {"get_path": lambda self, k: str(tmp_path) + "/"})()
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"db_a": ds_a, "db_b": ds_b}
        ds.df_test = df_test
        ds._build_test_ds_df()

        assert set(ds.test_ds_df.keys()) == {"db_a", "db_b"}
        assert len(ds.test_ds_df["db_a"]) == 1
        assert len(ds.test_ds_df["db_b"]) == 1


class _FakeDataset:
    """Minimal dataset stub with pre-built splits."""

    is_labeled = True
    got_gender = False
    got_age = False
    got_speaker = False
    name = "fake_db"

    def __init__(self, train_labels, test_labels, dev_labels=None):
        idx_tr = _make_segmented_index(
            [f"/data/tr_{i}.wav" for i in range(len(train_labels))]
        )
        idx_te = _make_segmented_index(
            [f"/data/te_{i}.wav" for i in range(len(test_labels))]
        )
        self.df_train = _tag_df(pd.DataFrame({"emotion": train_labels}, index=idx_tr))
        self.df_test = _tag_df(pd.DataFrame({"emotion": test_labels}, index=idx_te))
        self.df_train.is_labeled = True
        self.df_test.is_labeled = True
        if dev_labels is not None:
            idx_dev = _make_segmented_index(
                [f"/data/dev_{i}.wav" for i in range(len(dev_labels))]
            )
            self.df_dev = _tag_df(pd.DataFrame({"emotion": dev_labels}, index=idx_dev))
            self.df_dev.is_labeled = True

    def split(self):
        pass

    def split_3(self):
        pass

    def prepare_labels(self):
        pass


def _make_ds(datasets, split3=False):
    """Build a Datasplitter with real Util whose error() is captured."""
    from nkululeko.utils.util import Util

    ds = Datasplitter.__new__(Datasplitter)
    ds.util = Util("datasplitter")
    errors = []
    ds.util.error = lambda m: errors.append(m)
    ds.target = "emotion"
    ds.split3 = split3
    ds.got_speaker = False
    ds.datasets = datasets
    return ds, errors


class TestLabelEncoderUnseenLabels:
    """Verify fill_train_and_tests surfaces helpful errors for unseen labels."""

    def test_unseen_test_labels_calls_util_error(self):
        """fill_train_and_tests should report labels in test split not seen in training."""
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad"],
            test_labels=["angry"],
        )
        ds, errors = _make_ds({"db": fake_ds})

        ds.fill_train_and_tests()

        assert len(errors) == 1
        assert "angry" in errors[0]
        assert "not seen in training" in errors[0]

    def test_unseen_dev_labels_calls_util_error(self):
        """fill_train_and_tests should report unseen dev-split labels when split3=True."""
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad"],
            test_labels=["happy"],  # no unseen in test
            dev_labels=["angry"],  # unseen in dev
        )
        ds, errors = _make_ds({"db": fake_ds}, split3=True)

        ds.fill_train_and_tests()

        assert len(errors) == 1
        assert "angry" in errors[0]
        assert "not seen in training" in errors[0]

    def test_unseen_test_labels_with_nan_does_not_crash(self):
        """A partially-labeled test split (containing NaN) must not raise TypeError."""
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad"],
            test_labels=["angry", float("nan")],
        )
        ds, errors = _make_ds({"db": fake_ds})

        ds.fill_train_and_tests()

        assert len(errors) == 1
        assert "angry" in errors[0]
        assert "nan" not in errors[0]


class TestClassLabelBackup:
    """Regression tests for issue #46: class_label must be consistently created."""

    def test_class_label_backed_up_even_when_prepare_labels_skips_it(self):
        """Some code paths (e.g. a dataset whose prepare_labels() doesn't create
        class_label, as this fake dataset's no-op does) must still end up
        with class_label present after fill_train_and_tests(), so the
        original string labels remain recoverable after LabelEncoder
        transforms self.target in place.
        """
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad", "happy"],
            test_labels=["happy", "sad"],
        )
        ds, errors = _make_ds({"db": fake_ds})

        df_train, df_test = ds.fill_train_and_tests()

        assert errors == []
        assert "class_label" in df_train.columns
        assert "class_label" in df_test.columns
        assert list(df_train["class_label"]) == ["happy", "sad", "happy"]
        assert list(df_test["class_label"]) == ["happy", "sad"]
        # target column itself should now be integer-encoded, not the string
        assert df_train["emotion"].dtype.kind in "iu"

    def test_class_label_backed_up_for_dev_split(self):
        """The dev split must also get class_label."""
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad"],
            test_labels=["happy"],
            dev_labels=["sad", "happy"],
        )
        ds, errors = _make_ds({"db": fake_ds}, split3=True)

        df_train, df_test, df_dev = ds.fill_train_and_tests()

        assert errors == []
        assert "class_label" in df_dev.columns
        assert list(df_dev["class_label"]) == ["sad", "happy"]

    def test_unlabeled_test_split_gets_dummy_class_label_without_crashing(self):
        """An unlabeled test split (is_labeled=False) is filled with a random
        placeholder target via _add_random_target(); the reassignment goes
        through .astype("str"), which returns a new DataFrame and therefore
        drops the ad-hoc is_labeled attribute unless explicitly preserved.
        This must not raise AttributeError, and the resulting placeholder
        target should still be backed up into class_label.
        """
        fake_ds = _FakeDataset(
            train_labels=["happy", "sad"],
            test_labels=["happy"],
        )
        # A genuinely unlabeled test split: no "emotion" column, and the
        # is_labeled flag (read off the dataset object during aggregation,
        # not off df_test) must be False so the dummy-target branch is taken.
        fake_ds.is_labeled = False
        idx = _make_segmented_index(["/data/te_unlabeled_0.wav"])
        # Needs a non-target column so the DataFrame isn't considered
        # pandas-empty (which would skip the branch entirely via the
        # `if not self.df_test.empty:` guard) while still lacking "emotion".
        fake_ds.df_test = _tag_df(pd.DataFrame({"speaker": ["spk1"]}, index=idx))
        ds, errors = _make_ds({"db": fake_ds})

        df_train, df_test = ds.fill_train_and_tests()

        assert errors == []
        assert "class_label" in df_test.columns


class TestFillTrainAndTestsConcatenation:
    def test_multiple_datasets_concatenated_correctly(self, tmp_path, monkeypatch):
        """fill_train_and_tests should concat all datasets without O(n²) intermediate copies."""

        class FakeDataset:
            def __init__(self, name, train_files, test_files):
                self.name = name
                idx_tr = _make_segmented_index(train_files)
                idx_te = _make_segmented_index(test_files)
                self.df_train = _tag_df(pd.DataFrame(index=idx_tr))
                self.df_test = _tag_df(pd.DataFrame(index=idx_te))

            def split(self):
                pass  # already split in __init__

            def prepare_labels(self):
                pass

        monkeypatch.setitem(glob_conf.config["DATA"], "target", "none")
        monkeypatch.setattr(glob_conf, "target", None)

        ds_a = FakeDataset("a", ["/a/tr_1.wav", "/a/tr_2.wav"], ["/a/te_1.wav"])
        ds_b = FakeDataset("b", ["/b/tr_1.wav"], ["/b/te_1.wav", "/b/te_2.wav"])

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = _make_fake_util(tmp_path)
        ds.target = None
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"a": ds_a, "b": ds_b}

        df_train, df_test = ds.fill_train_and_tests()
        assert len(df_train) == 3  # 2 from a + 1 from b
        assert len(df_test) == 3  # 1 from a + 2 from b

    def test_flag_aggregation_any_wins_on_self_and_splits(self, tmp_path, monkeypatch):
        """Flags should be any-wins aggregated onto self and every split DataFrame."""

        class FakeDataset:
            def __init__(self, name, train_files, test_files, **flags):
                self.name = name
                idx_tr = _make_segmented_index(train_files)
                idx_te = _make_segmented_index(test_files)
                self.df_train = _tag_df(pd.DataFrame(index=idx_tr))
                self.df_test = _tag_df(pd.DataFrame(index=idx_te))
                for flag, value in flags.items():
                    setattr(self, flag, value)

            def split(self):
                pass  # already split in __init__

            def prepare_labels(self):
                pass

        monkeypatch.setitem(glob_conf.config["DATA"], "target", "none")
        monkeypatch.setattr(glob_conf, "target", None)

        # ds_a has every flag False, ds_b has every flag True: any-wins means
        # the aggregated result must be True on self and every split.
        ds_a = FakeDataset(
            "a",
            ["/a/tr_1.wav"],
            ["/a/te_1.wav"],
            is_labeled=False,
            got_gender=False,
            got_age=False,
            got_speaker=False,
        )
        ds_b = FakeDataset(
            "b",
            ["/b/tr_1.wav"],
            ["/b/te_1.wav"],
            is_labeled=True,
            got_gender=True,
            got_age=True,
            got_speaker=True,
        )

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = _make_fake_util(tmp_path)
        ds.target = None
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"a": ds_a, "b": ds_b}

        df_train, df_test = ds.fill_train_and_tests()

        for flag in ("is_labeled", "got_gender", "got_age", "got_speaker"):
            assert getattr(ds, flag) is True
            assert getattr(df_train, flag) is True
            assert getattr(df_test, flag) is True


class TestFillTrainAndTestsEarlyReturn:
    def test_unsupervised_returns_splits_without_labels(self, tmp_path, monkeypatch):
        """fill_train_and_tests returns (df_train, df_test) even when target is None."""

        class FakeDataset:
            def split(self):
                self.df_train = _tag_df(pd.DataFrame(index=range(2)))
                self.df_test = _tag_df(pd.DataFrame(index=range(1)))

            def prepare_labels(self):
                pass  # no-op: unsupervised run has no labels to encode

            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            name = "fake"

        monkeypatch.setitem(glob_conf.config["DATA"], "target", "none")
        monkeypatch.setattr(glob_conf, "target", None)

        fake_ds = FakeDataset()
        ds = Datasplitter.__new__(Datasplitter)
        ds.util = _make_fake_util(tmp_path)
        ds.target = None
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"fake": fake_ds}

        result = ds.fill_train_and_tests()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestUnseenLabelsError:
    """Test that unseen labels in test/dev sets produce helpful error messages."""

    _FLAGS = (
        "is_labeled",
        "is_test",
        "is_train",
        "is_val",
        "got_gender",
        "got_age",
        "got_speaker",
    )

    @staticmethod
    def _tag_df(df):
        """Set the standard split flags on a DataFrame in-place."""
        df.is_labeled = True
        df.got_gender = False
        df.got_speaker = False

    def _make_util(self, tmp_path, errors):
        """Return a mock Util with working copy_flags and error capturing."""
        flags = self._FLAGS

        class MockUtil:
            def get_path(self, k):
                return str(tmp_path) + "/"

            def config_val(self, sec, key, default):
                return default

            def debug(self, m):
                pass

            def warn(self, m):
                pass

            def error(self, m):
                errors.append(m)

            def copy_flags(self, src, tgt):
                for flag in flags:
                    if hasattr(src, flag):
                        setattr(tgt, flag, getattr(src, flag))

            def exp_is_classification(self):
                return True

        return MockUtil()

    def test_unseen_label_in_test_set(self, tmp_path):
        """fill_train_and_tests emits a descriptive error when test labels are unseen."""
        errors = []
        tag = self._tag_df

        class FakeDataset:
            name = "fake_db"
            is_labeled = True
            got_gender = False
            got_speaker = False

            def split(self):
                self.df_train = pd.DataFrame({"emotion": ["happy", "sad"]})
                tag(self.df_train)
                self.df_test = pd.DataFrame({"emotion": ["happy", "unknown"]})
                tag(self.df_test)

            def prepare_labels(self):
                pass

            df_train = pd.DataFrame()
            df_test = pd.DataFrame()

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = self._make_util(tmp_path, errors)
        ds.target = "emotion"
        ds.split3 = False
        ds.got_speaker = False
        ds.datasets = {"fake_db": FakeDataset()}

        ds.fill_train_and_tests()

        assert len(errors) == 1
        assert "unknown" in errors[0]
        assert "not seen in training" in errors[0]
        assert "Training labels are" in errors[0]

    def test_unseen_label_in_dev_set(self, tmp_path):
        """fill_train_and_tests emits a descriptive error when dev labels are unseen."""
        errors = []
        tag = self._tag_df

        class FakeDataset:
            name = "fake_db"
            is_labeled = True
            got_gender = False
            got_speaker = False

            def split_3(self):
                self.df_train = pd.DataFrame({"emotion": ["happy", "sad"]})
                tag(self.df_train)
                self.df_test = pd.DataFrame({"emotion": ["happy", "sad"]})
                tag(self.df_test)
                self.df_dev = pd.DataFrame({"emotion": ["happy", "novelcat"]})
                tag(self.df_dev)

            def prepare_labels(self):
                pass

            df_train = pd.DataFrame()
            df_test = pd.DataFrame()
            df_dev = pd.DataFrame()

        ds = Datasplitter.__new__(Datasplitter)
        ds.util = self._make_util(tmp_path, errors)
        ds.target = "emotion"
        ds.split3 = True
        ds.got_speaker = False
        ds.datasets = {"fake_db": FakeDataset()}

        ds.fill_train_and_tests()

        assert len(errors) == 1
        assert "novelcat" in errors[0]
        assert "not seen in training" in errors[0]
