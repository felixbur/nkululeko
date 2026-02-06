from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_emotion import EmotionPredictor


class TestEmotionPredictor:
    @patch("nkululeko.autopredict.ap_emotion.glob_conf")
    def test_init(self, mock_glob_conf):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = EmotionPredictor(df)
        
        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_emotion.FeatureExtractor")
    @patch("nkululeko.autopredict.ap_emotion.glob_conf")
    def test_predict(self, mock_glob_conf, mock_feature_extractor):
        mock_glob_conf.config = {"DATA": {"databases": "['test_db']"}}
        
        # Create mock dataframe
        mock_emotion_df = pd.DataFrame({"feat1": [0.1, 0.2, 0.3]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_emotion_df
        mock_feature_extractor.return_value = mock_extractor
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = EmotionPredictor(df)
        
        result = predictor.predict("train")
        
        assert "emotion_pred" in result.columns
        assert len(result) == 3
        # Currently returns "neutral" for all samples
        assert all(result["emotion_pred"] == "neutral")
