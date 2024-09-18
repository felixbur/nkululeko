""""
A predictor for biological sex.
Currently based on audEERING's agender model.
"""

import ast

import nkululeko.glob_conf as glob_conf
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.utils.util import Util


class GenderPredictor:
    """
    GenderPredictor
    predicting gender with the audEERING agender model

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("genderPredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting gender for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["agender_agender"], feats_name, split_selection
        )
        agender_df = self.feature_extractor.extract()
        pred_gender = agender_df.drop("age", axis=1).idxmax(axis=1)
        return_df = self.df.copy()
        return_df["gender_pred"] = pred_gender
        return return_df
