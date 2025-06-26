"""
A predictor for emotion classification.
Uses emotion2vec models for emotion prediction.
"""

import ast

import nkululeko.glob_conf as glob_conf
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
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        
        self.feature_extractor = FeatureExtractor(
            self.df, ["emotion2vec-large"], feats_name, split_selection
        )
        emotion_df = self.feature_extractor.extract()
        
        pred_emotion = ["neutral"] * len(emotion_df)
        
        return_df = self.df.copy()
        return_df["emotion_pred"] = pred_emotion
        return return_df
