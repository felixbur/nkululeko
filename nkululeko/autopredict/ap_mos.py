""""
A predictor for MOS - mean opinion score.
"""

import ast

import numpy as np

import nkululeko.glob_conf as glob_conf
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.utils.util import Util


class MOSPredictor:
    """
    MOSPredictor
    predicting MOS

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("mosPredictor")

    def predict(self, split_selection):
        self.util.debug(f"estimating MOS for {split_selection} samples")
        return_df = self.df.copy()
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["mos"], feats_name, split_selection
        )
        result_df = self.feature_extractor.extract()
        # replace missing values by 0
        result_df = result_df.fillna(0)
        result_df = result_df.replace(np.nan, 0)
        result_df.replace([np.inf, -np.inf], 0, inplace=True)
        pred_snr = result_df.mos * 100
        return_df["mos_pred"] = pred_snr.astype("int") / 100
        return return_df
