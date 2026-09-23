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
        """The predicted emotion is whichever class scores highest per row,
        from emotion2vec_emotion's own classification output (issue: this
        used to discard the extracted features and always return
        'neutral')."""
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        mock_emotion_df = pd.DataFrame(
            {
                "angry": [0.7, 0.1, 0.0],
                "happy": [0.2, 0.8, 0.0],
                "neutral": [0.1, 0.1, 0.0],
            }
        )
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_emotion_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = EmotionPredictor(df)

            result = predictor.predict("train")

        assert "emotion_pred" in result.columns
        assert len(result) == 3
        assert list(result["emotion_pred"]) == ["angry", "happy", "unknown"]

        # Verify it requests emotion2vec's classification feats_type, not
        # the raw-embedding one.
        args, _ = mock_feature_extractor.call_args
        assert args[1] == ["emotion2vec_emotion"]

    @patch("nkululeko.autopredict.ap_emotion.FeatureExtractor")
    def test_predict_all_rows_failed_extraction(self, mock_feature_extractor):
        """When every row fails extraction, Emotion2vec_emotion.extract()
        returns a DataFrame with zero columns (pd.DataFrame(all-empty-dicts)
        has no columns to infer); idxmax(axis=1) raises ValueError on that
        instead of reaching the per-row "unknown" handling -- this must be
        guarded before calling idxmax, not just handled per-row after."""
        context = ExperimentContext(config={"DATA": {"databases": "['test_db']"}})

        mock_emotion_df = pd.DataFrame(index=[0, 1, 2])
        assert mock_emotion_df.shape[1] == 0
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_emotion_df
        mock_feature_extractor.return_value = mock_extractor

        df = pd.DataFrame({"dummy": [1, 2, 3]})
        with use_context(context):
            predictor = EmotionPredictor(df)
            result = predictor.predict("train")

        assert list(result["emotion_pred"]) == ["unknown", "unknown", "unknown"]
