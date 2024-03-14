""""
A predictor for emotional valence.
Currently based on audEERING's emotional dimension model.
"""
from nkululeko.utils.util import Util
from nkululeko.feature_extractor import FeatureExtractor
import ast
import nkululeko.glob_conf as glob_conf


class ValencePredictor:
    """
    ValencePredictor
    predicting valence with the audEERING emotional dimension model

    """

    def __init__(self, df):
        self.df = df
        self.util = Util("valencePredictor")

    def predict(self, split_selection):
        self.util.debug(f"predicting valence for {split_selection} samples")
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feature_extractor = FeatureExtractor(
            self.df, ["auddim"], feats_name, split_selection
        )
        pred_df = self.feature_extractor.extract()
        pred_vals = pred_df.valence * 1000
        return_df = self.df.copy()
        return_df["valence_pred"] = pred_vals.astype("int") / 1000

        return return_df
