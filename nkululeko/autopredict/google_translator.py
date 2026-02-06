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

    async def translate_texts(self, texts: list[str]) -> list[str]:
        """Translate a list of texts using a single Translator session."""
        async with Translator() as translator:
            tasks = [translator.translate(text, dest="en") for text in texts]
            results = await asyncio.gather(*tasks)
            translations = [result.text for result in results]
        return translations

    def translate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate the text in the given DataFrame.

        :param df: DataFrame whose index contains tuples of (file, start, end).
        :return: DataFrame with translations indexed by the original index.
        :rtype: pd.DataFrame
        """
        file_name = ""
        seg_index = 0
        translations = [""] * len(df)
        translator_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), "translations")
        )

        uncached_positions = []
        uncached_texts = []
        uncached_cache_paths = []
        uncached_meta = []

        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            file = idx[0]
            start = idx[1]
            end = idx[2]
            if file != file_name:
                file_name = file
                seg_index = 0
            cache_name = audeer.basename_wo_ext(file) + str(seg_index)
            cache_path = audeer.path(translator_cache, cache_name + ".json")
            if os.path.isfile(cache_path):
                translations[i] = self.util.read_json(cache_path)["translation"]
            else:
                uncached_positions.append(i)
                uncached_texts.append(row["text"])
                uncached_cache_paths.append(cache_path)
                uncached_meta.append((file, start, end))
            seg_index += 1

        if uncached_texts:
            uncached_translations = asyncio.run(self.translate_texts(uncached_texts))
            for i, translation, cache_path, meta in zip(
                uncached_positions,
                uncached_translations,
                uncached_cache_paths,
                uncached_meta,
            ):
                file, start, end = meta
                translations[i] = translation
                self.util.save_json(
                    cache_path,
                    {
                        "translation": translation,
                        "file": file,
                        "start": start.total_seconds(),
                        "end": end.total_seconds(),
                    },
                )

        df = pd.DataFrame({self.language: translations}, index=df.index)
        return df
