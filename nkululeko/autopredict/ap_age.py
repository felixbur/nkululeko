""""
A predictor for age.
Currently based on audEERING's agender model.
"""
from nkululeko.utils.util import Util
from nkululeko.feature_extractor import FeatureExtractor
import ast
import nkululeko.glob_conf as glob_conf


class AgePredictor:
    """
    AgePredictor
    predicting age with the audEERING agender model

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("agePredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting age for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["agender_agender"], feats_name, split_selection
        )
        agender_df = self.feature_extractor.extract()
        pred_age = agender_df.age * 100
        #        pred_gender = agender_df.drop('age', axis=1).idxmax(axis=1)
        return_df = self.df.copy()
        #        return_df['gender_pred'] = pred_gender
        return_df["age_pred"] = pred_age.astype("int")
        return return_df
