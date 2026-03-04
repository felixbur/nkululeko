from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.autopredict.ap_mos import MOSPredictor


class TestMOSPredictor:
    @patch("nkululeko.autopredict.ap_mos.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = MOSPredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_mos.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_mos.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with MOS values
        mock_mos_df = pd.DataFrame({"mos": [3.5, 4.2, 2.8]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_mos_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = MOSPredictor(df)
        
        result = predictor.predict("train")
        
        assert "mos_pred" in result.columns
        assert len(result) == 3
        # Values should be multiplied by 100, cast to int, then divided by 100
        assert result["mos_pred"].iloc[0] == pytest.approx(3.5, abs=0.01)
        assert result["mos_pred"].iloc[1] == pytest.approx(4.2, abs=0.01)

    @patch("nkululeko.autopredict.ap_mos.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_mos.glob_conf")
    def test_predict_handles_nan_and_inf(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with NaN and inf values
        mock_mos_df = pd.DataFrame({"mos": [3.5, np.nan, np.inf, -np.inf]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_mos_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3, 4]})
        predictor = MOSPredictor(df)
        
        result = predictor.predict("train")
        
        assert "mos_pred" in result.columns
        # NaN and inf values should be replaced with 0
        assert result["mos_pred"].iloc[1] == 0.0
        assert result["mos_pred"].iloc[2] == 0.0
        assert result["mos_pred"].iloc[3] == 0.0
