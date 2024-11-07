"""seg_pyannote.py.

Segment a dataset with the Pyannote segmenter.
Also adds speaker ids to the segments.

"""

import pandas as pd
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm

import audformat
from audformat import segmented_index

from nkululeko.utils.util import Util


SAMPLING_RATE = 16000


class Pyannote_segmenter:
    def __init__(self, not_testing=True):
        # initialize the VAD model
        torch.set_num_threads(1)
        self.no_testing = not_testing
        self.util = Util("pyannote_segmenter")
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
        if device == "cpu":
            self.util.warn(
                "running pyannote on CPU can be really slow, consider using a GPU"
            )
        self.pipeline.to(torch.device(device))

    def get_segmentation_simple(self, file):

        annotation = self.pipeline(file[0])

        speakers, starts, ends, files = [], [], [], []
        # print the result
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            speakers.append(speaker)
            starts.append(start)
            files.append(file[0])
            ends.append(end)
        seg_index = segmented_index(files, starts, ends)
        return seg_index, speakers

    def get_segmentation(self, file, min_length, max_length):
        annotation = self.pipeline(file)
        files, starts, ends, speakers = [], [], [], []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = turn.start
            end = turn.end
            new_end = end
            handled = False
            while end - start > max_length:
                new_end = start + max_length
                if end - new_end < min_length:
                    new_end = end
                files.append(file[0])
                starts.append(start)
                ends.append(new_end)
                speakers.append(speaker)
                start += max_length
                handled = True
            if not handled and end - start > min_length:
                files.append(file[0])
                starts.append(start)
                ends.append(end)
                speakers.append(speaker)
        seg_index = segmented_index(files, starts, ends)
        return seg_index, speakers

    def segment_dataframe(self, df):
        dfs = []
        max_length = eval(self.util.config_val("SEGMENT", "max_length", "False"))
        if max_length:
            if self.no_testing:
                min_length = float(self.util.config_val("SEGMENT", "min_length", 2))
            else:
                min_length = 2
            self.util.debug(f"segmenting with max length: {max_length+min_length}")
        for file, values in tqdm(df.iterrows()):
            if max_length:
                index, speakers = self.get_segmentation(file, min_length, max_length)
            else:
                index, speakers = self.get_segmentation_simple(file)
            df = pd.DataFrame(
                values.to_dict(),
                index,
            )
            df["speaker"] = speakers
            dfs.append(df)
        return audformat.utils.concat(dfs)


def main():
    files = pd.Series(["test_wavs/very_long.wav"])
    df_sample = pd.DataFrame(index=files)
    df_sample["target"] = "anger"
    df_sample.index = audformat.utils.to_segmented_index(
        df_sample.index, allow_nat=False
    )
    segmenter = Pyannote_segmenter(not_testing=False)
    df_seg = segmenter.segment_dataframe(df_sample)

    def calc_dur(x):
        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()

    df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
    print(df_seg.head(100))


if __name__ == "__main__":
    main()
