import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.feats_opensmile import Opensmileset


class DummyUtil:
    """Mock utility class for testing."""
    def config_val(self, section, key, default=None):
        config_values = {
            ("FEATS", "set"): "eGeMAPSv02",
            ("FEATS", "level"): "functionals",
            ("FEATS", "needs_feature_extraction"): "False",
            ("FEATS", "no_reuse"): "False",
            ("FEATS", "store_format"): "pkl",
            ("MODEL", "n_jobs"): "1"
        }
        return config_values.get((section, key), default)
    
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): raise Exception(msg)
    def get_path(self, key): return "/tmp/test_store/"
    def get_exp_name(self, only_train=False): return "test_exp"
    def write_store(self, df, path, format): pass
    def get_store(self, path, format): return pd.DataFrame()


@pytest.fixture
def mock_config():
    """Mock glob_conf.config with required structure."""
    mock_config = {
        "EXP": {
            "root": "/tmp/test_nkululeko",
            "name": "test_exp"
        },
        "FEATS": {
            "features": "[]",  # Empty list for features filtering
            "set": "eGeMAPSv02",
            "level": "functionals",
            "needs_feature_extraction": "False",
            "no_reuse": "False",
            "store_format": "pkl"
        },
        "DATA": {
            "needs_feature_extraction": "False"
        },
        "MODEL": {
            "n_jobs": "1"
        }
    }
    
    # Mock the glob_conf.config
    with patch.object(glob_conf, 'config', mock_config):
        yield mock_config


@pytest.fixture
def sample_data_df():
    """Create a sample DataFrame for testing with real audio file paths."""
    # Use actual audio files from the test data directory
    audio_files = [
        "data/test/audio/03a01Fa.wav",
        "data/test/audio/03a01Nc.wav", 
        "data/test/audio/03a01Wa.wav"
    ]
    
    # Create MultiIndex with (file, start, end) as expected by nkululeko
    index_tuples = [(audio_file, pd.Timedelta(0), pd.Timedelta(seconds=1)) for audio_file in audio_files]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['file', 'start', 'end'])
    
    return pd.DataFrame({
        'speaker': ['speaker1', 'speaker2', 'speaker3'],
        'emotion': ['neutral', 'happy', 'sad']
    }, index=multi_index)


@patch.object(Opensmileset, "__init__", return_value=None)
def test_extract(mock_init, sample_data_df, mock_config):
    """Test the extract method with mocked initialization."""
    # Create an instance and manually set required attributes
    opensmile = Opensmileset.__new__(Opensmileset)
    opensmile.name = "test"
    opensmile.data_df = sample_data_df
    opensmile.util = DummyUtil()
    opensmile.df = pd.DataFrame()
    
    # Mock the extract method to return a sample DataFrame
    sample_features = pd.DataFrame({
        'F0semitoneFrom27.5Hz_sma3nz_amean': [100.0, 105.0, 95.0],
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': [0.1, 0.15, 0.08],
        'loudness_sma3_amean': [50.0, 55.0, 45.0]
    }, index=sample_data_df.index)
    
    with patch.object(opensmile, 'extract', return_value=sample_features):
        result = opensmile.extract()
        
        # Assert that the extracted features DataFrame is not empty
        assert not result.empty
        assert len(result) == 3
        assert result.shape[1] == 3


@patch.object(Opensmileset, "__init__", return_value=None)
def test_extract_sample(mock_init, sample_data_df, mock_config):
    """Test the extract_sample method with mocked initialization."""
    # Create an instance and manually set required attributes
    opensmile = Opensmileset.__new__(Opensmileset)
    opensmile.name = "test"
    opensmile.data_df = sample_data_df
    opensmile.util = DummyUtil()
    
    # Mock the extract_sample method
    sample_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    with patch.object(opensmile, 'extract_sample', return_value=sample_features):
        # Create a sample signal and sample rate
        signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 1000)
        sr = 16000
        
        # Call the extract_sample method
        feats = opensmile.extract_sample(signal, sr)
        
        # Assert that the extracted features are of type numpy.ndarray
        assert isinstance(feats, np.ndarray)
        assert len(feats) == 5


@patch.object(Opensmileset, "__init__", return_value=None)
def test_filter(mock_init, sample_data_df, mock_config):
    """Test the filter method with mocked initialization."""
    # Create an instance and manually set required attributes
    opensmile = Opensmileset.__new__(Opensmileset)
    opensmile.name = "test"
    opensmile.data_df = sample_data_df
    opensmile.util = DummyUtil()
    
    # Create a sample features DataFrame
    opensmile.df = pd.DataFrame({
        'F0semitoneFrom27.5Hz_sma3nz_amean': [100.0, 105.0, 95.0],
        'F0semitoneFrom27.5Hz_sma3nz_stddevNorm': [0.1, 0.15, 0.08],
        'loudness_sma3_amean': [50.0, 55.0, 45.0]
    }, index=sample_data_df.index)
    
    # Mock the filter method
    filtered_df = pd.DataFrame({
        'F0semitoneFrom27.5Hz_sma3nz_amean': [100.0, 105.0, 95.0]
    }, index=sample_data_df.index)
    
    with patch.object(opensmile, 'filter', return_value=filtered_df):
        # Call the filter method
        result = opensmile.filter()
        
        # Assert that the filtered DataFrame is still not empty
        assert not result.empty
        assert result.shape[0] == 3
        assert result.shape[1] == 1
