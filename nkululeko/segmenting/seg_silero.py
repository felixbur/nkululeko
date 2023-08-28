"""
seg_silero.py

segment a dataset with the Silero segmenter

"""

import torch
import audformat
from audformat.utils import to_filewise_index
from audformat import segmented_index
import pandas as pd 


vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False)

class Silero_segmenter:

    def __init__(self):
        # initialize the VAD model
        torch.set_num_threads(1)


    def get_segmentation(self, file):
    #    print(f'segmenting {file[0]}')
        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = vad_utils
        SAMPLING_RATE = 16000
        print('.', end='', flush=True)
        wav = read_audio(file[0], sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE)
        files, starts, ends = [], [], []
        for entry in speech_timestamps:
            start = float(entry['start']/10000.0)
            end = float(entry['end']/10000.0)
            files.append(file[0])
            starts.append(start)
            ends.append(end)
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
    files = pd.Series(['test_wavs/test_for_segment_2.wav'])
    df_sample = pd.DataFrame(index = files)
    df_sample['target'] = 'anger'
    df_sample.index = audformat.utils.to_segmented_index(df_sample.index, allow_nat=False)
    df_sample.head(10)
    segmenter = Silero_segmenter()
    df_seg = segmenter.segment_dataframe(df_sample)
    def calc_dur(x):
        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()
    df_seg['duration'] = df_seg.index.to_series().map(lambda x:calc_dur(x)) 
    print(df_seg.head())

if __name__ == '__main__':
    main()