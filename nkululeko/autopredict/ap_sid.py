""" "
A predictor for sid - Speaker ID.
"""

import numpy as np
from pyannote.audio import Pipeline
import torch

import audiofile

from nkululeko.feature_extractor import FeatureExtractor
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.files import concat_files
from nkululeko.utils.util import Util


class SIDPredictor:
    """SIDPredictor.

    predicting speaker id.
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("sidPredictor")
        hf_token = self.util.config_val("MODEL", "hf_token", None)
        if hf_token is None:
            self.util.error(
                "speaker id prediction needs huggingface token: [MODEL][hf_token]"
            )
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        device = self.util.config_val("MODEL", "device", "cpu")
        self.pipeline.to(torch.device(device))

    def predict(self, split_selection):
        self.util.debug(f"estimating speaker id for {split_selection} samples")
        return_df = self.df.copy()
        # 1) concat all audio files
        tmp_file = "tmp.wav"
        concat_files(return_df.index, tmp_file)
        # 2) get segmentations with pyannote
        sname = "pyannotation"
        if self.util.exist_pickle(sname):
            annotation = self.util.from_pickle(sname)
        else:
            annotation = self.pipeline(tmp_file)
            self.util.to_pickle(annotation, sname)

        speakers, starts, ends = [], [], []
        # print the result
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            speakers.append(speaker)
            starts.append(start)
            ends.append(end)
        # 3) map pyannote segments with orginal ones and assign speaker id
        target_speakers = []
        position = 0
        for idx, (file, start, end) in enumerate(return_df.index.to_list()):
            seg_start = start.total_seconds()
            seg_end = end.total_seconds()
            # file_duration = audiofile.duration(file)
            seg_duration = seg_end - seg_start
            offset = position + seg_start + seg_duration / 2
            l = [i < offset for i in starts]
            r = [i for i, x in enumerate(l) if x]
            s_index = r.pop()
            # self.util.debug(f"offset: {offset}, speaker = {speakers[s_index]}")
            position += seg_duration
            target_speakers.append(speakers[s_index])
        return_df["speaker"] = target_speakers
        return return_df
