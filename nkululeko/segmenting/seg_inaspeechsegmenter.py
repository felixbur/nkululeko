import warnings

import audformat
import pandas as pd
from audformat import segmented_index

# segment the data
from inaSpeechSegmenter import Segmenter


class Ina_segmenter:
    def __init__(self):
        # initialize the VAD model
        self.seg = Segmenter()
        warnings.simplefilter(action="ignore", category=FutureWarning)

    def get_segmentation(self, file):
        print(f"segmenting {file[0]}")
        segmentation = self.seg(file[0])
        files, starts, ends = [], [], []
        number = 0
        for entry in segmentation:
            kind = entry[0]
            start = entry[1]
            end = entry[2]
            if kind == "female" or kind == "male":
                files.append(file[0])
                starts.append(start)
                ends.append(end)
                number += 1
        print(f"found {number} segments")
        seg_index = segmented_index(files, starts, ends)
        return seg_index

    def segment_dataframe(self, df):
        dfs = []
        for file, values in df.iterrows():
            index = self.get_segmentation(file)
            dfs.append(
                pd.DataFrame(
                    values.to_dict(),
                    index,
                )
            )
        return audformat.utils.concat(dfs)


def main():
    files = pd.Series(["test_wavs/test_for_segment_2.wav"])
    df_sample = pd.DataFrame(index=files)
    df_sample["target"] = "anger"
    df_sample.index = audformat.utils.to_segmented_index(
        df_sample.index, allow_nat=False
    )
    df_sample.head(10)
    segmenter = Ina_segmenter()
    df_seg = segmenter.segment_dataframe(df_sample)

    def calc_dur(x):
        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()

    df_seg["duration"] = df_seg.index.to_series().map(lambda x: calc_dur(x))
    print(df_seg.head())


if __name__ == "__main__":
    main()
