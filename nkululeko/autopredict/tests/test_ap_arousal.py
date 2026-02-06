from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_arousal import ArousalPredictor


class TestArousalPredictor:
    @patch("nkululeko.autopredict.ap_arousal.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = ArousalPredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_arousal.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_arousal.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with arousal values
        mock_arousal_df = pd.DataFrame({"arousal": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_arousal_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = ArousalPredictor(df)
        
        result = predictor.predict("train")
        
        assert "arousal_pred" in result.columns
        assert len(result) == 3
        # Values should be multiplied by 1000, cast to int, then divided by 1000
        assert result["arousal_pred"].iloc[0] == pytest.approx(0.25, abs=0.001)
        assert result["arousal_pred"].iloc[1] == pytest.approx(0.45, abs=0.001)
        assert result["arousal_pred"].iloc[2] == pytest.approx(0.70, abs=0.001)
