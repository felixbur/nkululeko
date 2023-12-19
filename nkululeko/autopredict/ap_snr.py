""""
A predictor for SNR - signal-to-noise ratio.
"""
from nkululeko.utils.util import Util
import ast
import nkululeko.glob_conf as glob_conf
from nkululeko.feature_extractor import FeatureExtractor
import numpy as np


class SNRPredictor:
    """
    SNRPredictor
    predicting snr

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("snrPredictor")

    def predict(self, split_selection):
        self.util.debug(f"estimating SNR for {split_selection} samples")
        return_df = self.df.copy()
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["snr"], feats_name, split_selection
        )
        result_df = self.feature_extractor.extract()
        # replace missing values by 0
        result_df = result_df.fillna(0)
        result_df = result_df.replace(np.nan, 0)
        result_df.replace([np.inf, -np.inf], 0, inplace=True)
        pred_snr = result_df.snr * 100
        return_df["snr_pred"] = pred_snr.astype("int") / 100
        return return_df
