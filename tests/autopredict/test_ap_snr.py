from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.autopredict.ap_snr import SNRPredictor


class TestSNRPredictor:
    @patch("nkululeko.autopredict.ap_snr.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = SNRPredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_snr.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_snr.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with SNR values
        mock_snr_df = pd.DataFrame({"snr": [15.5, 20.2, 10.8]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_snr_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = SNRPredictor(df)
        
        result = predictor.predict("train")
        
        assert "snr_pred" in result.columns
        assert len(result) == 3
        assert result["snr_pred"].iloc[0] == pytest.approx(15.5, abs=0.01)

    @patch("nkululeko.autopredict.ap_snr.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_snr.glob_conf")
    def test_predict_handles_nan_and_inf(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        mock_snr_df = pd.DataFrame({"snr": [15.5, np.nan, np.inf, -np.inf]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_snr_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3, 4]})
        predictor = SNRPredictor(df)
        
        result = predictor.predict("train")
        
        assert result["snr_pred"].iloc[1] == 0.0
        assert result["snr_pred"].iloc[2] == 0.0
        assert result["snr_pred"].iloc[3] == 0.0
