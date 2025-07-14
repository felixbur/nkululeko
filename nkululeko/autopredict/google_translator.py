import os

import pandas as pd
import torch
from tqdm import tqdm

import asyncio
from googletrans import Translator

import audeer
import audiofile

from nkululeko.utils.util import Util

import httpx

class GoogleTranslator:
    def __init__(self, language="en", util=None):
        self.language = language
        self.util = util

    async def translate_text(self, text):
        async with Translator() as translator:
            result = translator.translate(text, dest="en")
            return (await result).text

    def translate_index(self, df:pd.DataFrame) ->  pd.DataFrame:
        """Transcribe the audio files in the given index.

        :param index: Index containing tuples of (file, start, end).
        :return: DataFrame with transcriptions indexed by the original index.
        :rtype: pd.DataFrame
        """
        file_name = ""
        seg_index = 0
        translations = []
        translator_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), "translations"))
        file_name = ""
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            file = idx[0]
            start = idx[1]
            end = idx[2]
            if file != file_name:
                file_name = file
                seg_index = 0
            cache_name = audeer.basename_wo_ext(file)+str(seg_index)
            cache_path = audeer.path(translator_cache, cache_name + ".json")
            if os.path.isfile(cache_path):
                translation = self.util.read_json(cache_path)["translation"]
            else:
                text = row['text']
                translation = asyncio.run(self.translate_text(text))
                self.util.save_json(cache_path, 
                                {"translation": translation, 
                                 "file": file, 
                                 "start": start.total_seconds(), 
                                 "end": end.total_seconds()})
            translations.append(translation)
            seg_index += 1

        df = pd.DataFrame({self.language:translations}, index=df.index)
        return df
