""""
A predictor for sid - Speaker ID.
"""

from pyannote.audio import Pipeline


import numpy as np

import nkululeko.glob_conf as glob_conf
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.utils.util import Util


class SIDPredictor:
    """SIDPredictor.

    predicting speaker id.
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("sidPredictor")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE",
        )

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
