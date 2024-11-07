""""
A predictor for sid - Speaker ID.
"""

import numpy as np
from pyannote.audio import Pipeline
import torch

from nkululeko.feature_extractor import FeatureExtractor
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


class SIDPredictor:
    """SIDPredictor.

    predicting speaker id.
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("sidPredictor")
        hf_token = self.util.config_val("Model", "hf_token", None)
        if hf_token is None:
            self.util.error(
                "speaker id prediction needs huggingface token: [MODEL][hf_token]"
            )
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        device = self.util.config_val("Model", "device", "cpu")
        self.pipeline.to(torch.device(device))

    def predict(self, split_selection):
        self.util.debug(f"estimating speaker id for {split_selection} samples")
        return_df = self.df.copy()
        # @todo
        # 1) concat all audio files
        # 2) get segmentations with pyannote
        # 3) map pyannote segments with orginal ones and assign speaker id

        return return_df

    def concat_files(self, df):
        pass
        # todo
        # please use https://audeering.github.io/audiofile/usage.html#read-a-file
