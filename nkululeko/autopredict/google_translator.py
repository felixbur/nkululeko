import asyncio
import logging
import os
from collections.abc import Iterable

import pandas as pd
from tqdm import tqdm

from googletrans import Translator

import audeer

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 10


class GoogleTranslator:
    def __init__(self, language="en", util=None):
        self.language = language
        self.util = util

    async def translate_text(self, text):
        async with Translator() as translator:
            result = translator.translate(text, dest=self.language)
            return (await result).text

    async def translate_texts(self, texts: Iterable[str]) -> list[str]:
        """Translate a list of texts using a single Translator session.

        Args:
            texts: Iterable of strings to translate.

        Returns:
            List of translated strings. Failed items are returned as empty
            strings.
        """
        texts = list(texts)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _translate_one(translator, text):
            async with semaphore:
                return await translator.translate(text, dest=self.language)

        async with Translator() as translator:
            tasks = [_translate_one(translator, text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        translations = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("Translation failed for item %d: %s", i, result)
                translations.append("")
            else:
                translations.append(result.text)
        return translations

    def translate_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate the text in the given DataFrame.

        Args:
            df: DataFrame whose index contains tuples of (file, start, end).

        Returns:
            DataFrame with translations indexed by the original index.
        """
        translations = [""] * len(df)
        translator_cache = audeer.mkdir(
            audeer.path(self.util.get_path("cache"), "translations", self.language)
        )

        uncached_positions = []
        uncached_texts = []
        uncached_cache_paths = []
        uncached_meta = []

        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            file = idx[0]
            start = idx[1]
            end = idx[2]
            start_ms = int(start.total_seconds() * 1000)
            end_ms = int(end.total_seconds() * 1000)
            cache_name = f"{audeer.basename_wo_ext(file)}_{start_ms}_{end_ms}"
            cache_path = audeer.path(translator_cache, cache_name + ".json")
            if os.path.isfile(cache_path):
                cached = self.util.read_json(cache_path)
                if cached.get("language") == self.language:
                    translations[i] = cached["translation"]
                    continue
            uncached_positions.append(i)
            uncached_texts.append(row["text"])
            uncached_cache_paths.append(cache_path)
            uncached_meta.append((file, start, end))

        if uncached_texts:
            try:
                uncached_translations = asyncio.run(
                    self.translate_texts(uncached_texts)
                )
            except RuntimeError:
                # A running event loop exists (e.g., inside Jupyter / async context).
                # Run the coroutine in a separate thread with its own event loop to
                # avoid the "cannot be called when another event loop is running" error.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run, self.translate_texts(uncached_texts)
                    )
                    uncached_translations = future.result()
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
                        "language": self.language,
                        "file": file,
                        "start": start.total_seconds(),
                        "end": end.total_seconds(),
                    },
                )

        df = pd.DataFrame({self.language: translations}, index=df.index)
        return df
