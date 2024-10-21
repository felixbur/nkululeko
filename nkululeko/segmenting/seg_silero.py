"""
seg_silero.py

segment a dataset with the Silero segmenter

"""

import audformat
import pandas as pd
import torch
from audformat import segmented_index
from tqdm import tqdm

from nkululeko.utils.util import Util

# from nkululeko.constants import SAMPLING_RATE

SAMPLING_RATE = 16000

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)


class Silero_segmenter:
    def __init__(self, not_testing=True):
        # initialize the VAD model
        torch.set_num_threads(1)
        self.no_testing = not_testing
        self.util = Util(has_config=not_testing)

    def get_segmentation_simple(self, file):
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = vad_utils
        SAMPLING_RATE = 16000
        wav = read_audio(file[0], sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=SAMPLING_RATE
        )
        files, starts, ends = [], [], []
        for entry in speech_timestamps:
            start = float(entry["start"] / SAMPLING_RATE)
            end = float(entry["end"] / SAMPLING_RATE)
            files.append(file[0])
            starts.append(start)
            ends.append(end)
        seg_index = segmented_index(files, starts, ends)
        return seg_index

    def get_segmentation(self, file, min_length, max_length):
        (
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks,
        ) = vad_utils
        SAMPLING_RATE = 16000
        wav = read_audio(file[0], sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=SAMPLING_RATE
        )
        files, starts, ends = [], [], []
        for entry in speech_timestamps:
            start = float(entry["start"] / SAMPLING_RATE)
            end = float(entry["end"] / SAMPLING_RATE)
            new_end = end
            handled = False
            while end - start > max_length:
                new_end = start + max_length
                if end - new_end < min_length:
                    new_end = end
                files.append(file[0])
                starts.append(start)
                ends.append(new_end)
                start += max_length
                handled = True
            if not handled and end - start > min_length:
                files.append(file[0])
                starts.append(start)
                ends.append(end)
        seg_index = segmented_index(files, starts, ends)
        return seg_index

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
                index = self.get_segmentation(file, min_length, max_length)
            else:
                index = self.get_segmentation_simple(file)
            dfs.append(
                pd.DataFrame(
                    values.to_dict(),
                    index,
                )
            )
        return audformat.utils.concat(dfs)


def main():
    #    files = pd.Series(['test_wavs/47_CF61_2_7.wav'])
    files = pd.Series(["test_wavs/very_long.wav"])
    df_sample = pd.DataFrame(index=files)
    df_sample["target"] = "anger"
    df_sample.index = audformat.utils.to_segmented_index(
        df_sample.index, allow_nat=False
    )
    df_sample.head(10)
    segmenter = Silero_segmenter(not_testing=False)
    df_seg = segmenter.segment_dataframe(df_sample)

    def calc_dur(x):
        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()

    df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
    print(df_seg.head(100))


if __name__ == "__main__":
    main()
