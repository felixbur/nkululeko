from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_age import AgePredictor


class TestAgePredictor:
    @patch("nkululeko.autopredict.ap_age.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = AgePredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_age.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_age.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with age values
        mock_age_df = pd.DataFrame({"age": [0.25, 0.45, 0.70]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_age_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = AgePredictor(df)
        
        result = predictor.predict("train")
        
        assert "age_pred" in result.columns
        assert len(result) == 3
        assert result["age_pred"].dtype == int
        # Age is multiplied by 100 and cast to int
        assert list(result["age_pred"]) == [25, 45, 70]
