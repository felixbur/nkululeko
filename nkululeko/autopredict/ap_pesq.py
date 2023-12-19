""""
A predictor for PESQ - Perceptual Evaluation of Speech Quality.
"""
from nkululeko.utils.util import Util
import ast
import nkululeko.glob_conf as glob_conf
from nkululeko.feature_extractor import FeatureExtractor
import numpy as np


class PESQPredictor:
    """
    PESQPredictor
    predicting PESQ

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("pesqPredictor")

    def predict(self, split_selection):
        self.util.debug(f"estimating PESQ for {split_selection} samples")
        return_df = self.df.copy()
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["squim"], feats_name, split_selection
        )
        result_df = self.feature_extractor.extract()
        # replace missing values by 0
        result_df = result_df.fillna(0)
        result_df = result_df.replace(np.nan, 0)
        result_df.replace([np.inf, -np.inf], 0, inplace=True)
        pred_vals = result_df.pesq * 100
        return_df["pesq_pred"] = pred_vals.astype("int") / 100
        return return_df
