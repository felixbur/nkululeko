"""A predictor for text.

Currently based on whisper model.
"""

import ast

import torch

from nkululeko.feature_extractor import FeatureExtractor
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


class TextPredictor:
    """TextPredictor.

    predicting text with the whisper model
    """

    def __init__(self, df, util=None):
        self.df = df
        if util is not None:
            self.util = util
        else:
            # create a new util instance
            # this is needed to access the config and other utilities
            # in the autopredict module
            self.util = Util("textPredictor")
        from nkululeko.autopredict.whisper_transcriber import Transcriber
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self.util.config_val("MODEL", "device", device)
        self.transcriber = Transcriber(
            device=device,
            language=self.util.config_val("EXP", "language", "en"),
            util=self.util,
        )
    def predict(self, split_selection):
        self.util.debug(f"predicting text for {split_selection} samples")
        df = self.transcriber.transcribe_index(
            self.df.index
        )
        return_df = self.df.copy()
        return_df["text"] = df["text"].values
        return return_df
