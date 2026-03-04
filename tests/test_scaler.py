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
