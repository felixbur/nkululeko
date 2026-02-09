from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_dominance import DominancePredictor


class TestDominancePredictor:
    @patch("nkululeko.autopredict.ap_dominance.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = DominancePredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_dominance.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_dominance.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with dominance values
        mock_dominance_df = pd.DataFrame({"dominance": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_dominance_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = DominancePredictor(df)
        
        result = predictor.predict("train")
        
        assert "dominance_pred" in result.columns
        assert len(result) == 3
        # Values should be multiplied by 1000, cast to int, then divided by 1000
        assert result["dominance_pred"].iloc[0] == pytest.approx(0.25, abs=0.001)
