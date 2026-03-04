from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_valence import ValencePredictor


class TestValencePredictor:
    @patch("nkululeko.autopredict.ap_valence.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = ValencePredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_valence.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_valence.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with valence values
        mock_valence_df = pd.DataFrame({"valence": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_valence_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = ValencePredictor(df)
        
        result = predictor.predict("train")
        
        assert "valence_pred" in result.columns
        assert len(result) == 3
        # Values should be multiplied by 1000, cast to int, then divided by 1000
        assert result["valence_pred"].iloc[0] == pytest.approx(0.25, abs=0.001)
