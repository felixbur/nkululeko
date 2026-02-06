from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_gender import GenderPredictor


class TestGenderPredictor:
    @patch("nkululeko.autopredict.ap_gender.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = GenderPredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_gender.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_gender.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe with age and gender columns
        mock_agender_df = pd.DataFrame({
            "age": [0.25, 0.45, 0.70],
            "male": [0.8, 0.2, 0.6],
            "female": [0.2, 0.8, 0.4]
        })
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_agender_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = GenderPredictor(df)
        
        result = predictor.predict("train")
        
        assert "gender_pred" in result.columns
        assert len(result) == 3
        # Check that gender is predicted based on max value
        assert result["gender_pred"].iloc[0] == "male"
        assert result["gender_pred"].iloc[1] == "female"
        assert result["gender_pred"].iloc[2] == "male"
