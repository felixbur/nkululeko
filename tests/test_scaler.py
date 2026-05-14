import pytest
import pandas as pd
import numpy as np
from nkululeko.scaler import Scaler


class DummyUtil:
    def __init__(self, name):
        pass

    def error(self, msg):
        raise ValueError(msg)

    def debug(self, msg):
        pass


# Patch Util to avoid side effects
import nkululeko.scaler

nkululeko.scaler.Util = DummyUtil


@pytest.fixture
def sample_data():
    train_feats = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]}, index=[0, 1, 2]
    )
    test_feats = pd.DataFrame({"f1": [2.0, 3.0], "f2": [5.0, 7.0]}, index=[3, 4])
    dev_feats = pd.DataFrame({"f1": [1.5, 2.5], "f2": [4.5, 6.5]}, index=[5, 6])
    train_data_df = pd.DataFrame({"speaker": ["a", "b", "c"]}, index=[0, 1, 2])
    test_data_df = pd.DataFrame({"speaker": ["a", "b"]}, index=[3, 4])
    dev_data_df = pd.DataFrame({"speaker": ["a", "c"]}, index=[5, 6])
    return train_data_df, test_data_df, dev_data_df, train_feats, test_feats, dev_feats


@pytest.mark.parametrize(
    "scaler_type",
    [
        "standard",
        "robust",
        "minmax",
        "maxabs",
        "normalizer",
        "powertransformer",
        "quantiletransformer",
    ],
)
def test_scale_all_basic(scaler_type, sample_data):
    train_data_df, test_data_df, dev_data_df, train_feats, test_feats, dev_feats = (
        sample_data
    )
    scaler = Scaler(
        train_data_df,
        test_data_df,
        train_feats.copy(),
        test_feats.copy(),
        scaler_type,
        dev_x=dev_data_df,
        dev_y=dev_feats.copy(),
    )
    train_scaled, dev_scaled, test_scaled = scaler.scale_all()
    # Check output shapes
    assert train_scaled.shape == train_feats.shape
    assert test_scaled.shape == test_feats.shape
    assert dev_scaled.shape == dev_feats.shape
    # Check that scaling changed the data (except for normalizer, which may not change all-zero rows)
    assert not np.allclose(train_scaled.values, train_feats.values)


def test_scale_all_no_dev(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "standard"
    )
    train_scaled, test_scaled = scaler.scale_all()
    assert train_scaled.shape == train_feats.shape
    assert test_scaled.shape == test_feats.shape


def test_scale_all_bins(sample_data):
    train_data_df, test_data_df, dev_data_df, train_feats, test_feats, dev_feats = (
        sample_data
    )
    scaler = Scaler(
        train_data_df,
        test_data_df,
        train_feats.copy(),
        test_feats.copy(),
        "bins",
        dev_x=dev_data_df,
        dev_y=dev_feats.copy(),
    )
    train_scaled, dev_scaled, test_scaled = scaler.scale_all()
    # Should be strings '0', '0.5', or '1'
    for df in [train_scaled, dev_scaled, test_scaled]:
        for col in df.columns:
            assert set(df[col].unique()).issubset({"0", "0.5", "1"})


def test_scale_all_unknown_scaler(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    with pytest.raises(ValueError):
        Scaler(
            train_data_df, test_data_df, train_feats, test_feats, "unknown"
        ).scale_all()


# ---------------------------------------------------------------------------
# scale() dispatcher
# ---------------------------------------------------------------------------


def test_scale_non_speaker_calls_scale_all(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "standard"
    )
    train_scaled, test_scaled = scaler.scale()
    assert train_scaled.shape == train_feats.shape
    assert test_scaled.shape == test_feats.shape


def test_scale_speaker_calls_speaker_scale(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "speaker"
    )
    result = scaler.scale()
    assert len(result) == 2
    assert result[0].shape == train_feats.shape
    assert result[1].shape == test_feats.shape


# ---------------------------------------------------------------------------
# scale_df()
# ---------------------------------------------------------------------------


def test_scale_df_preserves_index_and_columns(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "standard"
    )
    scaler.scaler.fit(train_feats.values)
    scaled = scaler.scale_df(test_feats.copy())
    assert list(scaled.index) == list(test_feats.index)
    assert list(scaled.columns) == list(test_feats.columns)


def test_scale_df_returns_dataframe(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "minmax"
    )
    scaler.scaler.fit(train_feats.values)
    scaled = scaler.scale_df(train_feats.copy())
    assert isinstance(scaled, pd.DataFrame)


def test_scale_df_values_differ_after_transform(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "standard"
    )
    scaler.scaler.fit(train_feats.values)
    scaled = scaler.scale_df(train_feats.copy())
    assert not np.allclose(scaled.values, train_feats.values)


# ---------------------------------------------------------------------------
# speaker_scale() and speaker_scale_df()
# ---------------------------------------------------------------------------


def test_speaker_scale_no_dev_returns_two(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(
        train_data_df, test_data_df, train_feats.copy(), test_feats.copy(), "speaker"
    )
    result = scaler.speaker_scale()
    assert len(result) == 2


def test_speaker_scale_with_dev_returns_three(sample_data):
    train_data_df, test_data_df, dev_data_df, train_feats, test_feats, dev_feats = (
        sample_data
    )
    scaler = Scaler(
        train_data_df,
        test_data_df,
        train_feats.copy(),
        test_feats.copy(),
        "speaker",
        dev_x=dev_data_df,
        dev_y=dev_feats.copy(),
    )
    result = scaler.speaker_scale()
    assert len(result) == 3


def test_speaker_scale_df_normalizes_per_speaker():
    train_data_df = pd.DataFrame({"speaker": ["a", "a", "b", "b"]}, index=[0, 1, 2, 3])
    feats = pd.DataFrame({"f1": [1.0, 3.0, 10.0, 20.0]}, index=[0, 1, 2, 3])
    dummy_train = pd.DataFrame({"f1": [0.0, 1.0]})
    dummy_test = pd.DataFrame({"f1": [0.0]})
    scaler = Scaler(train_data_df, train_data_df, dummy_train, dummy_test, "speaker")
    scaled = scaler.speaker_scale_df(train_data_df, feats.copy())
    # Each speaker's features should be independently standardized
    spk_a = scaled.loc[[0, 1], "f1"].values
    spk_b = scaled.loc[[2, 3], "f1"].values
    assert np.allclose(np.mean(spk_a), 0.0, atol=1e-10)
    assert np.allclose(np.mean(spk_b), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _bin()
# ---------------------------------------------------------------------------


def test_bin_low_values_become_zero(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(train_data_df, test_data_df, train_feats, test_feats, "bins")
    result = scaler._bin(np.array([0.0, 1.0]), b1=5.0, b2=9.0)
    assert result.tolist() == ["0", "0"]


def test_bin_middle_values_become_half(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(train_data_df, test_data_df, train_feats, test_feats, "bins")
    result = scaler._bin(np.array([6.0, 7.0]), b1=5.0, b2=9.0)
    assert result.tolist() == ["0.5", "0.5"]


def test_bin_high_values_become_one(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(train_data_df, test_data_df, train_feats, test_feats, "bins")
    result = scaler._bin(np.array([10.0, 20.0]), b1=5.0, b2=9.0)
    assert result.tolist() == ["1", "1"]


def test_bin_returns_series(sample_data):
    train_data_df, test_data_df, _, train_feats, test_feats, _ = sample_data
    scaler = Scaler(train_data_df, test_data_df, train_feats, test_feats, "bins")
    result = scaler._bin(np.array([1.0, 5.0, 10.0]), b1=3.0, b2=7.0)
    assert isinstance(result, pd.Series)
    assert set(result.tolist()).issubset({"0", "0.5", "1"})
