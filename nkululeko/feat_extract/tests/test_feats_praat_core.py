import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import parselmouth
import pytest
from scipy.stats import lognorm

from nkululeko.feat_extract.feats_praat_core import (AudioFeatureExtractor,
                                                     add_derived_features,
                                                     compute_features,
                                                     get_speech_rate, run_pca,
                                                     speech_rate)


class TestAudioFeatureExtractor:
    
    @pytest.fixture
    def extractor(self):
        return AudioFeatureExtractor(f0min=75, f0max=300)
    
    @pytest.fixture
    def mock_sound(self):
        sound = Mock()
        sound.get_total_duration.return_value = 2.5
        return sound
    
    def test_init(self):
        extractor = AudioFeatureExtractor(f0min=50, f0max=400)
        assert extractor.f0min == 50
        assert extractor.f0max == 400
    
    def test_init_default_values(self):
        extractor = AudioFeatureExtractor()
        assert extractor.f0min == 75
        assert extractor.f0max == 300
    
    @patch('nkululeko.feat_extract.feats_praat_core.call')
    def test_extract_pitch_features(self, mock_call, extractor, mock_sound):
        mock_pitch = Mock()
        mock_point_process = Mock()
        
        # Mock call return values
        mock_call.side_effect = [
            150.0,  # mean_f0
            25.0,   # stdev_f0
            Mock(), # harmonicity object
            0.8,    # hnr
            0.01,   # local_jitter
            0.05,   # localabsolute_jitter
            0.02,   # rap_jitter
            0.03,   # ppq5_jitter
            0.04,   # ddp_jitter
            0.1,    # local_shimmer
            0.5,    # localdb_shimmer
            0.15,   # apq3_shimmer
            0.2,    # apq5_shimmer
            0.25,   # apq11_shimmer
            0.3     # dda_shimmer
        ]
        
        result = extractor._extract_pitch_features(mock_sound, mock_pitch, mock_point_process)
        
        assert result['meanF0Hz'] == 150.0
        assert result['stdevF0Hz'] == 25.0
        assert result['HNR'] == 0.8
        assert result['localJitter'] == 0.01
        assert len(result) == 14
    
    @patch('nkululeko.feat_extract.feats_praat_core.call')
    def test_extract_formant_features(self, mock_call, extractor, mock_sound):
        mock_point_process = Mock()
        
        # Mock formant values
        mock_call.side_effect = [
            Mock(),  # formants object
            3,       # num_points
            0.5,     # time from index 1
            800.0,   # f1 at time 0.5
            1200.0,  # f2 at time 0.5
            2800.0,  # f3 at time 0.5
            3500.0,  # f4 at time 0.5
            1.0,     # time from index 2
            750.0,   # f1 at time 1.0
            1150.0,  # f2 at time 1.0
            2700.0,  # f3 at time 1.0
            3400.0,  # f4 at time 1.0
            1.5,     # time from index 3
            820.0,   # f1 at time 1.5
            1250.0,  # f2 at time 1.5
            2900.0,  # f3 at time 1.5
            3600.0   # f4 at time 1.5
        ]
        
        result = extractor._extract_formant_features(mock_sound, mock_point_process)
        
        assert 'f1_mean' in result
        assert 'f2_mean' in result
        assert 'f3_mean' in result
        assert 'f4_mean' in result
        assert 'f1_median' in result
        assert 'f2_median' in result
        assert 'f3_median' in result
        assert 'f4_median' in result
        assert len(result) == 8
    
    @patch('nkululeko.feat_extract.feats_praat_core.call')
    def test_extract_formant_features_with_nan(self, mock_call, extractor, mock_sound):
        mock_point_process = Mock()
        
        # Mock with some NaN values
        mock_call.side_effect = [
            Mock(),  # formants object
            2,       # num_points
            0.5,     # time from index 1
            float('nan'),  # f1 at time 0.5 (NaN)
            1200.0,  # f2 at time 0.5
            float('nan'),  # f3 at time 0.5 (NaN)
            3500.0,  # f4 at time 0.5
            1.0,     # time from index 2
            750.0,   # f1 at time 1.0
            1150.0,  # f2 at time 1.0
            2700.0,  # f3 at time 1.0
            float('nan')   # f4 at time 1.0 (NaN)
        ]
        
        result = extractor._extract_formant_features(mock_sound, mock_point_process)
        
        # Should handle NaN values gracefully
        assert 'f1_mean' in result
        assert not np.isnan(result['f2_mean'])
        assert len(result) == 8
    
    def test_calculate_pause_distribution_empty_list(self, extractor):
        result = extractor._calculate_pause_distribution([])
        
        assert np.isnan(result['pause_lognorm_mu'])
        assert np.isnan(result['pause_lognorm_sigma'])
        assert np.isnan(result['pause_lognorm_ks_pvalue'])
        assert np.isnan(result['pause_mean_duration'])
        assert np.isnan(result['pause_std_duration'])
        assert np.isnan(result['pause_cv'])
    
    def test_calculate_pause_distribution_valid_data(self, extractor):
        pause_durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = extractor._calculate_pause_distribution(pause_durations)
        
        assert not np.isnan(result['pause_mean_duration'])
        assert not np.isnan(result['pause_std_duration'])
        assert not np.isnan(result['pause_cv'])
        assert result['pause_mean_duration'] == 0.3
        assert len(result) == 6


class TestRunPCA:
    
    def test_run_pca_valid_data(self):
        # Create test dataframe with jitter and shimmer measures
        data = {
            'localJitter': [0.01, 0.02, 0.015],
            'localabsoluteJitter': [0.05, 0.06, 0.055],
            'rapJitter': [0.02, 0.03, 0.025],
            'ppq5Jitter': [0.03, 0.04, 0.035],
            'ddpJitter': [0.04, 0.05, 0.045],
            'localShimmer': [0.1, 0.2, 0.15],
            'localdbShimmer': [0.5, 0.6, 0.55],
            'apq3Shimmer': [0.15, 0.25, 0.2],
            'apq5Shimmer': [0.2, 0.3, 0.25],
            'apq11Shimmer': [0.25, 0.35, 0.3],
            'ddaShimmer': [0.3, 0.4, 0.35]
        }
        df = pd.DataFrame(data)
        
        result = run_pca(df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'JitterPCA' in result.columns
        assert 'ShimmerPCA' in result.columns
        assert len(result) == 3
    
    def test_run_pca_with_nan_values(self):
        # Create test dataframe with NaN values
        data = {
            'localJitter': [0.01, np.nan, 0.015],
            'localabsoluteJitter': [0.05, 0.06, np.nan],
            'rapJitter': [0.02, 0.03, 0.025],
            'ppq5Jitter': [0.03, 0.04, 0.035],
            'ddpJitter': [0.04, 0.05, 0.045],
            'localShimmer': [0.1, 0.2, 0.15],
            'localdbShimmer': [0.5, 0.6, 0.55],
            'apq3Shimmer': [0.15, 0.25, 0.2],
            'apq5Shimmer': [0.2, 0.3, 0.25],
            'apq11Shimmer': [0.25, 0.35, 0.3],
            'ddaShimmer': [0.3, 0.4, 0.35]
        }
        df = pd.DataFrame(data)
        
        result = run_pca(df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'JitterPCA' in result.columns
        assert 'ShimmerPCA' in result.columns
    
    def test_run_pca_single_file(self):
        # Test with single file (should handle ValueError)
        data = {
            'localJitter': [0.01],
            'localabsoluteJitter': [0.05],
            'rapJitter': [0.02],
            'ppq5Jitter': [0.03],
            'ddpJitter': [0.04],
            'localShimmer': [0.1],
            'localdbShimmer': [0.5],
            'apq3Shimmer': [0.15],
            'apq5Shimmer': [0.2],
            'apq11Shimmer': [0.25],
            'ddaShimmer': [0.3]
        }
        df = pd.DataFrame(data)
        
        result = run_pca(df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]['JitterPCA'] == 0
        assert result.iloc[0]['ShimmerPCA'] == 0


class TestAddDerivedFeatures:
    
    def test_add_derived_features(self):
        # Create test dataframe with required columns
        data = {
            'f1_median': [800, 750, 820],
            'f2_median': [1200, 1150, 1250],
            'f3_median': [2800, 2700, 2900],
            'f4_median': [3500, 3400, 3600],
            'localJitter': [0.01, 0.02, 0.015],
            'localabsoluteJitter': [0.05, 0.06, 0.055],
            'rapJitter': [0.02, 0.03, 0.025],
            'ppq5Jitter': [0.03, 0.04, 0.035],
            'ddpJitter': [0.04, 0.05, 0.045],
            'localShimmer': [0.1, 0.2, 0.15],
            'localdbShimmer': [0.5, 0.6, 0.55],
            'apq3Shimmer': [0.15, 0.25, 0.2],
            'apq5Shimmer': [0.2, 0.3, 0.25],
            'apq11Shimmer': [0.25, 0.35, 0.3],
            'ddaShimmer': [0.3, 0.4, 0.35]
        }
        df = pd.DataFrame(data)
        
        result = add_derived_features(df)
        
        # Check PCA columns are added
        assert 'JitterPCA' in result.columns
        assert 'ShimmerPCA' in result.columns
        
        # Check vocal tract features are added
        assert 'pF' in result.columns
        assert 'fdisp' in result.columns
        assert 'avgFormant' in result.columns
        assert 'mff' in result.columns
        assert 'fitch_vtl' in result.columns
        assert 'delta_f' in result.columns
        assert 'vtl_delta_f' in result.columns
    
    def test_add_derived_features_with_nan(self):
        # Test with NaN values
        data = {
            'f1_median': [np.nan, 750, 820],
            'f2_median': [1200, np.nan, 1250],
            'f3_median': [2800, 2700, np.nan],
            'f4_median': [3500, 3400, 3600],
            'localJitter': [0.01, 0.02, 0.015],
            'localabsoluteJitter': [0.05, 0.06, 0.055],
            'rapJitter': [0.02, 0.03, 0.025],
            'ppq5Jitter': [0.03, 0.04, 0.035],
            'ddpJitter': [0.04, 0.05, 0.045],
            'localShimmer': [0.1, 0.2, 0.15],
            'localdbShimmer': [0.5, 0.6, 0.55],
            'apq3Shimmer': [0.15, 0.25, 0.2],
            'apq5Shimmer': [0.2, 0.3, 0.25],
            'apq11Shimmer': [0.25, 0.35, 0.3],
            'ddaShimmer': [0.3, 0.4, 0.35]
        }
        df = pd.DataFrame(data)
        
        result = add_derived_features(df)
        
        # Should handle NaN values without raising errors
        assert 'pF' in result.columns
        assert 'fdisp' in result.columns
        assert len(result) == len(df)


class TestComputeFeatures:
    
    def test_compute_features_function_exists(self):
        # Simple test to verify the function exists and is importable
        assert callable(compute_features)


class TestSpeechRate:
    
    def test_speech_rate_function_exists(self):
        # Simple test to verify the function exists and is importable
        assert callable(speech_rate)


class TestGetSpeechRate:
    
    def test_get_speech_rate_function_exists(self):
        # Simple test to verify the function exists and is importable
        assert callable(get_speech_rate)


class TestPraatIntegration:
    """Integration tests for complete Praat feature extraction pipeline."""
    
    def test_compute_features_with_real_audio_file(self):
        """Test that all 45 features can be extracted from a real audio file."""
        import datetime
        import os

        # Use a real audio file from the test data
        audio_file = "./data/test/audio/debate_sample.wav"
        
        # Verify the test audio file exists
        assert os.path.exists(audio_file), f"Test audio file not found: {audio_file}"
        
        # Create a mock file index similar to what nkululeko uses
        # Format: (file_path, start_time, end_time)
        file_index = pd.DataFrame([
            (audio_file, datetime.timedelta(seconds=0), datetime.timedelta(seconds=5))
        ], columns=['file', 'start', 'end'])
        
        # Set the DataFrame index to match what compute_features expects
        file_index = file_index.set_index(['file', 'start', 'end']).index
        
        # Extract features using the main compute_features function
        features_df = compute_features(file_index)
        
        # Verify the result is a DataFrame
        assert isinstance(features_df, pd.DataFrame), "compute_features should return a DataFrame"
        
        # Verify we have exactly one row (one audio file)
        assert len(features_df) == 1, f"Expected 1 row, got {len(features_df)}"
        
        # Verify we have approximately 45 features (exact count may vary with optimizations)
        expected_min_features = 40  # Allow some tolerance
        expected_max_features = 50  # Allow some tolerance
        actual_features = len(features_df.columns)
        
        assert expected_min_features <= actual_features <= expected_max_features, \
            f"Expected ~45 features (range {expected_min_features}-{expected_max_features}), got {actual_features}. " \
            f"Features: {list(features_df.columns)}"
        
        # Verify that all expected core features are present
        expected_core_features = [
            'duration', 'meanF0Hz', 'stdevF0Hz', 'HNR',
            'f1_mean', 'f1_median', 'f2_mean', 'f2_median', 
            'f3_mean', 'f3_median', 'f4_mean', 'f4_median',
            'localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
            'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer',
            'JitterPCA', 'ShimmerPCA',  # From PCA
            'pF', 'fdisp', 'avgFormant', 'mff', 'fitch_vtl', 'delta_f', 'vtl_delta_f',  # Vocal tract
            'nsyll', 'npause', 'phonationtime_s', 'speechrate_nsyll_dur', 
            'articulation_rate_nsyll_phonationtime', 'ASD_speakingtime_nsyll',  # Speech rate
        ]
        
        missing_features = [feat for feat in expected_core_features if feat not in features_df.columns]
        assert len(missing_features) == 0, f"Missing expected features: {missing_features}"
        
        # Verify that most features are not NaN (allowing some tolerance for edge cases)
        non_nan_features = features_df.notna().sum(axis=1).iloc[0]
        total_features = len(features_df.columns)
        
        # At least 80% of features should be non-NaN for a valid audio file
        min_valid_features = int(0.8 * total_features)
        assert non_nan_features >= min_valid_features, \
            f"Too many NaN features: {non_nan_features}/{total_features} are valid, " \
            f"expected at least {min_valid_features}"
        
        # Verify that specific features have reasonable values
        row = features_df.iloc[0]
        
        # Duration should be positive and approximately 5 seconds (with some tolerance)
        assert 3.0 <= row['duration'] <= 7.0, f"Duration seems unreasonable: {row['duration']}"
        
        # F0 values should be in human speech range if detected
        if not pd.isna(row['meanF0Hz']):
            assert 50 <= row['meanF0Hz'] <= 500, f"Mean F0 seems unreasonable: {row['meanF0Hz']}"
        
        # Formant values should be in typical ranges if detected
        for i in range(1, 5):
            formant_mean = row[f'f{i}_mean']
            if not pd.isna(formant_mean):
                assert 200 <= formant_mean <= 4000, f"Formant F{i} mean seems unreasonable: {formant_mean}"
        
        print(f"SUCCESS: Extracted {actual_features} features from real audio file")
        print(f"Feature names: {list(features_df.columns)}")
        print(f"Non-NaN features: {non_nan_features}/{total_features}")
        
    def test_feature_extraction_robustness_multiple_files(self):
        """Test feature extraction with multiple real audio files."""
        import datetime
        import os

        # Test with multiple audio files
        audio_dir = "./data/test/audio"
        available_files = [
            "debate_sample.wav",
            "03a01Fa.wav", 
            "03a01Nc.wav"
        ]
        
        # Filter to only files that actually exist
        test_files = []
        for fname in available_files:
            fpath = os.path.join(audio_dir, fname)
            if os.path.exists(fpath):
                test_files.append(fpath)
        
        assert len(test_files) >= 1, "Need at least one test audio file"
        
        # Create file index for multiple files
        file_index_data = []
        for audio_file in test_files:
            file_index_data.append((audio_file, datetime.timedelta(seconds=0), datetime.timedelta(seconds=3)))
        
        file_index = pd.DataFrame(file_index_data, columns=['file', 'start', 'end'])
        file_index = file_index.set_index(['file', 'start', 'end']).index
        
        # Extract features
        features_df = compute_features(file_index)
        
        # Verify we have the correct number of rows
        assert len(features_df) == len(test_files), f"Expected {len(test_files)} rows, got {len(features_df)}"
        
        # Verify all files produced some valid features
        for i, test_file in enumerate(test_files):
            row = features_df.iloc[i]
            non_nan_count = row.notna().sum()
            total_features = len(features_df.columns)
            
            # Each file should have at least some valid features
            min_valid = int(0.5 * total_features)  # More lenient for multiple files
            assert non_nan_count >= min_valid, \
                f"File {test_file} has too few valid features: {non_nan_count}/{total_features}"
        
        print(f"SUCCESS: Extracted features from {len(test_files)} files")
        print(f"Total features per file: {len(features_df.columns)}")
        
    def test_expected_feature_count_matches_documentation(self):
        """Test that the actual feature count matches the documented count in the code."""
        import datetime
        import os
        
        audio_file = "./data/test/audio/debate_sample.wav"
        assert os.path.exists(audio_file), f"Test audio file not found: {audio_file}"
        
        file_index = pd.DataFrame([
            (audio_file, datetime.timedelta(seconds=0), datetime.timedelta(seconds=2))
        ], columns=['file', 'start', 'end'])
        file_index = file_index.set_index(['file', 'start', 'end']).index
        
        features_df = compute_features(file_index)
        actual_count = len(features_df.columns)
        
        # According to the docstring, we expect ~43-45 features
        # The exact count may vary based on optimization and implementation details
        expected_range = (42, 47)  # Allow some tolerance
        
        assert expected_range[0] <= actual_count <= expected_range[1], \
            f"Feature count {actual_count} is outside expected range {expected_range}. " \
            f"This may indicate changes to the feature extraction implementation."
        
        # Print the actual features for documentation/debugging
        feature_categories = {
            'basic': ['duration', 'meanF0Hz', 'stdevF0Hz', 'HNR'],
            'formants': [col for col in features_df.columns if col.startswith('f') and ('_mean' in col or '_median' in col)],
            'jitter': [col for col in features_df.columns if 'Jitter' in col],
            'shimmer': [col for col in features_df.columns if 'Shimmer' in col],
            'pca': [col for col in features_df.columns if 'PCA' in col],
            'vocal_tract': [col for col in features_df.columns if col in ['pF', 'fdisp', 'avgFormant', 'mff', 'fitch_vtl', 'delta_f', 'vtl_delta_f']],
            'speech_rate': [col for col in features_df.columns if col in ['nsyll', 'npause', 'phonationtime_s', 'speechrate_nsyll_dur', 'articulation_rate_nsyll_phonationtime', 'ASD_speakingtime_nsyll']],
            'pause_distribution': [col for col in features_df.columns if 'pause' in col.lower()],
            'other': []
        }
        
        # Classify all features
        classified_features = set()
        for category, features in feature_categories.items():
            classified_features.update(features)
        
        feature_categories['other'] = [col for col in features_df.columns if col not in classified_features]
        
        print(f"\nFeature breakdown (total: {actual_count}):")
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features - {features}")
        
        # Verify we have features in all major categories
        required_categories = ['basic', 'formants', 'jitter', 'shimmer']
        for category in required_categories:
            assert len(feature_categories[category]) > 0, f"No features found in {category} category"