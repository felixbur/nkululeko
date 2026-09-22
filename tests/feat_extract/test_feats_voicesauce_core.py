import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.feat_extract.feats_voicesauce_core import (
    VoiceSauceFeatureExtractor,
    compute_features,
)


class TestVoiceSauceFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return VoiceSauceFeatureExtractor(f0min=75, f0max=300)

    def test_init(self):
        extractor = VoiceSauceFeatureExtractor(f0min=50, f0max=400)
        assert extractor.f0min == 50
        assert extractor.f0max == 400

    def test_init_default_values(self):
        extractor = VoiceSauceFeatureExtractor()
        assert extractor.f0min == 75
        assert extractor.f0max == 300

    @patch("nkululeko.feat_extract.feats_voicesauce_core.call")
    def test_band_amplitude_db(self, mock_call, extractor):
        mock_call.return_value = 1e-4
        db = extractor._band_amplitude_db(Mock(), 200)
        assert db == pytest.approx(10 * np.log10(1e-4))

    @patch("nkululeko.feat_extract.feats_voicesauce_core.call")
    def test_band_amplitude_db_nonpositive_energy_is_nan(self, mock_call, extractor):
        mock_call.return_value = 0
        assert np.isnan(extractor._band_amplitude_db(Mock(), 200))

    @patch("nkululeko.feat_extract.feats_voicesauce_core.call")
    def test_band_amplitude_db_praat_error_is_nan(self, mock_call, extractor):
        import parselmouth

        mock_call.side_effect = parselmouth.PraatError("boom")
        assert np.isnan(extractor._band_amplitude_db(Mock(), 200))

    def test_extract_harmonic_amplitude_features_no_points(self, extractor):
        """With zero analysis points, every feature must come back NaN
        rather than raising (e.g. IndexError/statistics.mean on an empty
        list), the same way feats_praat_core handles no formant points."""
        sound = Mock()
        sound.get_total_duration.return_value = 1.0
        pitch = Mock()
        formants = Mock()
        point_process = Mock()

        with patch(
            "nkululeko.feat_extract.feats_voicesauce_core.call"
        ) as mock_call:
            mock_call.return_value = 0  # "Get number of points" -> 0

            result = extractor._extract_harmonic_amplitude_features(
                sound, pitch, formants, point_process
            )

        expected_keys = {"H1", "H2", "H4", "A1", "A2", "A3", "H1H2", "H1A1", "H1A2", "H1A3", "H2H4"}
        assert set(result.keys()) == expected_keys
        assert all(np.isnan(v) for v in result.values())

    @patch("nkululeko.feat_extract.feats_voicesauce_core.call")
    def test_extract_cpp_feature(self, mock_call, extractor):
        mock_call.side_effect = [Mock(), 12.34]
        result = extractor._extract_cpp_feature(Mock())
        assert result == {"CPP": 12.34}

    @patch("nkululeko.feat_extract.feats_voicesauce_core.call")
    def test_extract_cpp_feature_praat_error_is_nan(self, mock_call, extractor):
        import parselmouth

        mock_call.side_effect = parselmouth.PraatError("boom")
        result = extractor._extract_cpp_feature(Mock())
        assert np.isnan(result["CPP"])


class TestComputeFeatures:
    def test_compute_features_function_exists(self):
        assert callable(compute_features)


class TestVoiceSauceIntegration:
    """Integration tests using real audio, mirroring
    test_feats_praat_core.py's TestPraatIntegration."""

    def test_compute_features_with_real_audio_file(self):
        import datetime

        audio_file = "./data/test/audio/debate_sample.wav"
        assert os.path.exists(audio_file), f"Test audio file not found: {audio_file}"

        file_index = pd.DataFrame(
            [
                (
                    audio_file,
                    datetime.timedelta(seconds=0),
                    datetime.timedelta(seconds=5),
                )
            ],
            columns=["file", "start", "end"],
        )
        file_index = file_index.set_index(["file", "start", "end"]).index

        features_df = compute_features(file_index)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 1

        expected_features = [
            "H1",
            "H2",
            "H4",
            "A1",
            "A2",
            "A3",
            "H1H2",
            "H1A1",
            "H1A2",
            "H1A3",
            "H2H4",
            "CPP",
        ]
        assert list(features_df.columns) == expected_features

        row = features_df.iloc[0]
        non_nan = row.notna().sum()
        assert non_nan >= int(0.8 * len(expected_features)), (
            f"Too many NaN features: {non_nan}/{len(expected_features)} valid"
        )

    def test_feature_extraction_robustness_multiple_files(self):
        import datetime

        audio_dir = "./data/test/audio"
        available_files = ["debate_sample.wav", "03a01Fa.wav", "03a01Nc.wav"]
        test_files = [
            os.path.join(audio_dir, f)
            for f in available_files
            if os.path.exists(os.path.join(audio_dir, f))
        ]
        assert len(test_files) >= 1, "Need at least one test audio file"

        file_index_data = [
            (f, datetime.timedelta(seconds=0), datetime.timedelta(seconds=3))
            for f in test_files
        ]
        file_index = pd.DataFrame(file_index_data, columns=["file", "start", "end"])
        file_index = file_index.set_index(["file", "start", "end"]).index

        features_df = compute_features(file_index)

        assert len(features_df) == len(test_files)
        for i in range(len(test_files)):
            assert features_df.iloc[i].notna().sum() >= 1
