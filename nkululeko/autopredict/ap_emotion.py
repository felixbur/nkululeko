"""
A predictor for emotion classification.
Uses emotion2vec models for emotion prediction.
"""

import ast

from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.utils.util import Util


class EmotionPredictor:
    """
    EmotionPredictor
    predicting emotion with emotion2vec models
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("emotionPredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting emotion for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(self.util.config["DATA"]["databases"]))

        self.feature_extractor = FeatureExtractor(
            self.df,
            ["emotion2vec_emotion"],
            feats_name,
            split_selection,
            context=self.util.context,
        )
        # emotion2vec_emotion's columns are the emotion2vec_plus_* model's
        # own class scores (angry/happy/neutral/...); the predicted emotion
        # is whichever column scores highest per row.
        emotion_df = self.feature_extractor.extract()
        pred_emotion = emotion_df.idxmax(axis=1)
        # A row that failed extraction comes back all-zero (see
        # Emotion2vec_emotion._predict_one); idxmax would then arbitrarily
        # pick the first column as if it were a real (confident)
        # prediction, so mark those as unknown instead.
        pred_emotion = pred_emotion.where(emotion_df.sum(axis=1) > 0, "unknown")

        return_df = self.df.copy()
        return_df["emotion_pred"] = pred_emotion
        return return_df
