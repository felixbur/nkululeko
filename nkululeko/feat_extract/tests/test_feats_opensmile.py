import pytest
import numpy as np
import pandas as pd
import os

from nkululeko.feat_extract.feats_opensmile import Opensmileset


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


def test_extract(sample_data_df):
    """Test the extract method with real audio files."""
    # Create an instance of Opensmileset
    opensmile = Opensmileset('test', sample_data_df)
    
    # Call the extract method
    opensmile.extract()
    
    # Assert that the extracted features DataFrame is not empty
    assert not opensmile.df.empty
    assert len(opensmile.df) == 3
    # OpenSMILE features should have many columns (typically hundreds)
    assert opensmile.df.shape[1] > 10


def test_extract_sample(sample_data_df):
    """Test the extract_sample method."""
    # Create an instance of Opensmileset
    opensmile = Opensmileset('test', sample_data_df)
    
    # Create a sample signal and sample rate
    signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 1000)  # Make it longer to avoid warnings
    sr = 16000
    
    # Call the extract_sample method
    feats = opensmile.extract_sample(signal, sr)
    
    # Assert that the extracted features are of type numpy.ndarray
    assert isinstance(feats, np.ndarray)


def test_filter(sample_data_df):
    """Test the filter method."""
    # Create an instance of Opensmileset
    opensmile = Opensmileset('test', sample_data_df)
    
    # Call the extract method to populate the df attribute
    opensmile.extract()
    
    # Store the original shape
    original_shape = opensmile.df.shape
    
    # Call the filter method
    opensmile.filter()
    
    # Assert that the filtered DataFrame is still not empty
    # The exact shape depends on the filter implementation
    assert not opensmile.df.empty
    # The number of rows should remain the same (3 audio files)
    assert opensmile.df.shape[0] == 3
    # The filter may reduce the number of features, but we just check it's reasonable
    assert opensmile.df.shape[1] > 0
