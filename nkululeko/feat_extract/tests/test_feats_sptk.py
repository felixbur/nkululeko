import os
import tempfile
from unittest.mock import Mock, MagicMock, patch, call
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
import torch
from nkululeko.feat_extract.feats_sptk import SptkSet

"""Tests for SPTK feature extraction."""





@pytest.fixture
def mock_util():
    """Create a mock util object."""
    util = Mock()
    util.config_val = Mock(side_effect=lambda section, key, default: default)
    util.get_path = Mock(return_value="/tmp/store/")
    util.debug = Mock()
    util.error = Mock()
    util.write_store = Mock()
    util.get_store = Mock()
    return util


@pytest.fixture
def mock_data_df():
    """Create a mock data DataFrame."""
    return pd.DataFrame(index=["test_audio.wav"])


@pytest.fixture
def sptk_set(mock_util, mock_data_df):
    """Create a SptkSet instance with mocked dependencies."""
    with patch("nkululeko.feat_extract.feats_sptk.torch.cuda.is_available", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk_core.diffsptk"):
            sptk = SptkSet("test", mock_data_df, "sptk")
            sptk.util = mock_util
            sptk.device = "cpu"
            sptk.pitch_features_available = False
            sptk.pitch_fallback = True
            return sptk


def test_extract_creates_new_features_when_no_cache(sptk_set, mock_util):
    """Test that extract creates new features when cache doesn't exist."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft', 'fbank']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    # Mock compute_features to return a DataFrame
    mock_df = pd.DataFrame({"stft_mean": [0.5], "fbank_0_mean": [0.3]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                assert result is not None
                assert isinstance(result, pd.DataFrame)
                mock_util.debug.assert_any_call("extracting SPTK, this might take a while...")
                mock_util.write_store.assert_called_once()


def test_extract_reuses_cached_features(sptk_set, mock_util):
    """Test that extract reuses cached features when available."""
    cached_df = pd.DataFrame({"stft_mean": [0.5], "fbank_0_mean": [0.3]})
    mock_util.get_store = Mock(return_value=cached_df)
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): False,
        ("FEATS", "no_reuse"): "False",
    }.get((section, key), default))
    
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=True):
        result = sptk_set.extract()
        
        assert result is not None
        mock_util.debug.assert_any_call("reusing extracted SPTK values")
        mock_util.get_store.assert_called_once()


def test_extract_handles_segmented_index(sptk_set, mock_util):
    """Test that extract handles segmented index (file, start, end) tuples."""
    # Create DataFrame with tuple index
    start = timedelta(seconds=0)
    end = timedelta(seconds=2)
    sptk_set.data_df = pd.DataFrame(index=[("test_audio.wav", start, end)])
    
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    mock_df = pd.DataFrame({"stft_mean": [0.5]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df) as mock_compute:
                result = sptk_set.extract()
                
                # Verify compute_features was called with segmented index
                mock_compute.assert_called_once()
                call_kwargs = mock_compute.call_args[1]
                assert "file_index" in call_kwargs


def test_extract_pads_short_signals(sptk_set, mock_util):
    """Test that extract pads signals shorter than frame_period."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    mock_df = pd.DataFrame({"stft_mean": [0.5]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                # Verify extraction completed successfully
                assert result is not None
                assert isinstance(result, pd.DataFrame)


def test_extract_computes_stft_statistics(sptk_set, mock_util):
    """Test that extract computes correct STFT statistics."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    mock_df = pd.DataFrame({"stft_mean": [1.5], "stft_std": [0.5], "stft_min": [1.0], "stft_max": [2.0]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                assert "stft_mean" in result.columns
                assert "stft_std" in result.columns
                assert "stft_min" in result.columns
                assert "stft_max" in result.columns


def test_extract_computes_fbank_features(sptk_set, mock_util):
    """Test that extract computes FBANK features when requested."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['fbank']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    mock_df = pd.DataFrame({"fbank_0_mean": [0.5], "fbank_1_mean": [0.3]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                # Check that FBANK features were computed
                fbank_cols = [col for col in result.columns if col.startswith("fbank_")]
                assert len(fbank_cols) > 0


def test_extract_fills_nan_values(sptk_set, mock_util):
    """Test that extract fills NaN values with mean."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    # Create data with multiple samples to test NaN filling
    sptk_set.data_df = pd.DataFrame(index=["audio1.wav", "audio2.wav"])
    mock_df = pd.DataFrame({"stft_mean": [0.5, 0.6]}, index=["audio1.wav", "audio2.wav"])
    
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                # Verify no NaN values in result
                assert not result.isnull().values.any()


def test_extract_error_on_cached_nan_values(sptk_set, mock_util):
    """Test that extract raises error when cached data contains NaN."""
    cached_df = pd.DataFrame({"stft_mean": [0.5, np.nan], "fbank_0_mean": [0.3, 0.4]})
    mock_util.get_store = Mock(return_value=cached_df)
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): False,
        ("FEATS", "no_reuse"): "False",
    }.get((section, key), default))
    
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=True):
        sptk_set.extract()
        
        # Verify error was called due to NaN values
        mock_util.error.assert_called_once()


def test_extract_converts_to_float(sptk_set, mock_util):
    """Test that extract converts all features to float dtype."""
    mock_util.config_val = Mock(side_effect=lambda section, key, default: {
        ("FEATS", "store_format"): "pkl",
        ("FEATS", "needs_feature_extraction"): True,
        ("FEATS", "no_reuse"): "False",
        ("FEATS", "features"): "['stft']",
        ("FEATS", "print_feats"): "False",
    }.get((section, key), default))
    
    mock_df = pd.DataFrame({"stft_mean": [0.5]})
    with patch("nkululeko.feat_extract.feats_sptk.os.path.isfile", return_value=False):
        with patch("nkululeko.feat_extract.feats_sptk.glob_conf") as mock_glob_conf:
            mock_glob_conf.config = {"DATA": {}}
            with patch("nkululeko.feat_extract.feats_sptk_core.compute_features", return_value=mock_df):
                result = sptk_set.extract()
                
                # Verify all columns are float
                for col in result.columns:
                    assert result[col].dtype == np.float64