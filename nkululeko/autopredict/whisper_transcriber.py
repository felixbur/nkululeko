import os

import pandas as pd
import torch
from tqdm import tqdm
import whisper

import audeer
import audiofile

from nkululeko.utils.util import Util


class Transcriber:
    def __init__(self, model_name="turbo", device=None, language="en", util=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=device)
        self.language = language
        self.util = util

    def transcribe_file(self, audio_path):
        """Transcribe the audio file at the given path.

        :param audio_path: Path to the audio file to transcribe.
        :return: Transcription text.
        """
        result = self.model.transcribe(
            audio_path, language=self.language, without_timestamps=True)
        result = result["text"].strip()
        return result

    def transcribe_array(self, signal, sampling_rate):
        """Transcribe the audio file at the given path.

        :param audio_path: Path to the audio file to transcribe.
        :return: Transcription text.
        """
        tmporary_path = "temp.wav"
        audiofile.write(
            "temp.wav", signal, sampling_rate, format="wav")
        result = self.transcribe_file(tmporary_path)
        return result

    def transcribe_index(self, index:pd.Index) ->  pd.DataFrame:
        """Transcribe the audio files in the given index.

        :param index: Index containing tuples of (file, start, end).
        :return: DataFrame with transcriptions indexed by the original index.
        :rtype: pd.DataFrame
        """
        file_name = ""
        seg_index = 0
        transcriptions = []
        transcriber_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), "transcriptions"))
        for idx, (file, start, end) in enumerate(
            tqdm(index.to_list())
        ):
            if file != file_name:
                file_name = file
                seg_index = 0
            cache_name = audeer.basename_wo_ext(file)+str(seg_index)
            cache_path = audeer.path(transcriber_cache, cache_name + ".json")
            if os.path.isfile(cache_path):
                transcription = self.util.read_json(cache_path)["transcription"]
            else:
                dur = end.total_seconds() - start.total_seconds()
                y, sr = audiofile.read(file, offset=start, duration=dur)
                transcription = self.transcribe_array(
                    y, sr)
                self.util.save_json(cache_path, 
                                {"transcription": transcription, 
                                 "file": file, 
                                 "start": start.total_seconds(), 
                                 "end": end.total_seconds()})
            transcriptions.append(transcription)
            seg_index += 1

        df = pd.DataFrame({"text":transcriptions}, index=index)
        return df
