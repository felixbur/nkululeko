import sys
from types import SimpleNamespace

import numpy as np
import pytest

import nkululeko.feat_extract.feats_cqcc as feats_cqcc
from nkululeko.feat_extract.feats_cqcc import CqccFeatureExtractor
from nkululeko.feat_extract.feats_cqcc import CqccSet
from nkululeko.feat_extract.feats_lfcc import LfccFeatureExtractor
from nkululeko.feat_extract.feats_lfcc import LfccSet


class FakeSignal:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)

    def unsqueeze(self, dim):
        assert dim == 0
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeLfccOutput:
    def squeeze(self, dim):
        assert dim == 0
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])


class DummyLfccTransform:
    def __call__(self, signal_tensor):
        assert isinstance(signal_tensor, FakeSignal)
        return FakeLfccOutput()


def test_lfcc_extract_returns_mean_and_std():
    extractor = LfccFeatureExtractor.__new__(LfccFeatureExtractor)
    extractor.available = True
    extractor.transform = DummyLfccTransform()

    feats = extractor.extract(FakeSignal(np.ones(10)))

    assert feats["lfcc_0_mean"] == pytest.approx(2.0)
    assert feats["lfcc_0_std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert feats["lfcc_1_mean"] == pytest.approx(6.0)
    assert feats["lfcc_1_std"] == pytest.approx(np.std([4.0, 6.0, 8.0]))


def test_lfcc_extract_skips_when_unavailable(capsys):
    extractor = LfccFeatureExtractor.__new__(LfccFeatureExtractor)
    extractor.available = False
    extractor.warning = "lfcc unavailable"

    feats = extractor.extract(FakeSignal(np.ones(10)))

    assert feats == {}
    assert "lfcc unavailable" in capsys.readouterr().out


def test_cqcc_extract_returns_mean_and_std(monkeypatch):
    monkeypatch.setattr(
        feats_cqcc,
        "librosa",
        SimpleNamespace(cqt=lambda *args, **kwargs: np.ones((3, 2))),
        raising=False,
    )
    monkeypatch.setattr(
        feats_cqcc,
        "_scipy_dct",
        lambda *args, **kwargs: np.array([[1.0, 3.0], [2.0, 6.0]]),
        raising=False,
    )

    extractor = CqccFeatureExtractor(
        sample_rate=16000,
        frame_period=80,
        n_cqcc=2,
        n_cqt_bins=3,
    )
    extractor.available = True

    feats = extractor.extract(FakeSignal(np.ones(10)))

    assert feats["cqcc_0_mean"] == pytest.approx(2.0)
    assert feats["cqcc_0_std"] == pytest.approx(1.0)
    assert feats["cqcc_1_mean"] == pytest.approx(4.0)
    assert feats["cqcc_1_std"] == pytest.approx(2.0)


def test_cqcc_extract_skips_when_unavailable(capsys):
    extractor = CqccFeatureExtractor(sample_rate=16000, frame_period=80)
    extractor.available = False
    extractor.warning = "cqcc unavailable"

    feats = extractor.extract(FakeSignal(np.ones(10)))

    assert feats == {}
    assert "cqcc unavailable" in capsys.readouterr().out


class DummySptkSet:
    def __init__(self, name, data_df, feats_type):
        self.name = name
        self.data_df = data_df
        self.feats_type = feats_type
        self.features_requested = []
        self.df = None

    def extract(self):
        self.df = {"features": self.features_requested}
        return self.df

    def filter(self):
        return None

    def extract_sample(self, signal, sr):
        return self.features_requested


def test_lfcc_set_forces_lfcc_feature(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "nkululeko.feat_extract.feats_sptk",
        SimpleNamespace(SptkSet=DummySptkSet),
    )

    feat_set = LfccSet("test", None, "lfcc")

    assert feat_set._sptk.features_requested == ["lfcc"]
    assert feat_set.extract() == {"features": ["lfcc"]}
    assert feat_set.extract_sample(np.ones(10), 16000) == ["lfcc"]


def test_cqcc_set_forces_cqcc_feature(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "nkululeko.feat_extract.feats_sptk",
        SimpleNamespace(SptkSet=DummySptkSet),
    )

    feat_set = CqccSet("test", None, "cqcc")

    assert feat_set._sptk.features_requested == ["cqcc"]
    assert feat_set.extract() == {"features": ["cqcc"]}
    assert feat_set.extract_sample(np.ones(10), 16000) == ["cqcc"]


def test_feature_extractor_accepts_lfcc_and_cqcc():
    from nkululeko.feature_extractor import FeatureExtractor

    feature_extractor = FeatureExtractor(None, [], "test", "train")

    assert feature_extractor._get_feat_extractor_class("lfcc") is LfccSet
    assert feature_extractor._get_feat_extractor_class("cqcc") is CqccSet
