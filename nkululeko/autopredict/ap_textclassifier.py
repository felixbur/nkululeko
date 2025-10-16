""" "
A predictor for zero shot text classification.
"""

import ast
import pandas as pd
import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract.feats_textclassifier import TextClassifier
from nkululeko.utils.util import Util


class TextClassificationPredictor:
    """
    TextClassificationPredictor
    predicting text classes with a zero-shot classification model
    """

    def __init__(self, df, util=None):
        self.df = df
        if util is not None:
            self.util = util
        else:
            # create a new util instance
            # this is needed to access the config and other utilities
            # in the autopredict module
            self.util = Util("textClassifierPredictor")

    def predict(self, split_selection):
        self.util.debug(f"classifying text for {split_selection} samples")
        self.feature_extractor = TextClassifier("textclassifier", self.df)
        result_df = self.feature_extractor.extract()
        return_df = pd.concat([self.df, result_df], axis=1)
        return return_df
