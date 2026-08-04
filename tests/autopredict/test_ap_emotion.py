from unittest.mock import Mock, patch

import pandas as pd

from nkululeko.autopredict.ap_emotion import EmotionPredictor
from nkululeko.experiment_context import ExperimentContext, use_context


class TestEmotionPredictor:
    def test_init(self):
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = EmotionPredictor(df)

        assert predictor.df is df
        assert predictor.util is not None

    @patch("nkululeko.autopredict.ap_emotion.FeatureExtractor")
    def test_predict(self, mock_feature_extractor):
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        # Create mock dataframe
        mock_emotion_df = pd.DataFrame({"feat1": [0.1, 0.2, 0.3]})
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_emotion_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = EmotionPredictor(df)

            result = predictor.predict("train")

        assert "emotion_pred" in result.columns
        assert len(result) == 3
        # Currently returns "neutral" for all samples
        assert all(result["emotion_pred"] == "neutral")
