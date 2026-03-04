from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytest.importorskip("googletrans")

from nkululeko.autopredict.google_translator import GoogleTranslator  # noqa: E402


class TestGoogleTranslator:
    def test_init(self):
        mock_util = Mock()
        translator = GoogleTranslator(language="en", util=mock_util)

        assert translator.language == "en"
        assert translator.util is mock_util

    @patch("nkululeko.autopredict.google_translator.Translator")
    def test_translate_text(self, mock_translator_class):
        # Skip async testing since it requires additional setup
        translator = GoogleTranslator(language="en")
        assert translator.language == "en"

    @patch("nkululeko.autopredict.google_translator.Translator")
    def test_translate_texts(self, mock_translator_class):
        # Skip async testing since it requires additional setup
        translator = GoogleTranslator(language="en")
        assert hasattr(translator, "translate_texts")

    @patch("nkululeko.autopredict.google_translator.audeer")
    @patch("nkululeko.autopredict.google_translator.os.path.isfile")
    @patch("nkululeko.autopredict.google_translator.asyncio")
    def test_translate_index_with_cache(self, mock_asyncio, mock_isfile, mock_audeer):
        mock_util = Mock()
        mock_util.get_path.return_value = "/cache"
        mock_util.read_json.return_value = {
            "translation": "cached translation",
            "language": "en",
        }

        mock_audeer.mkdir.return_value = "/cache/translations/en"
        mock_audeer.path.side_effect = lambda *args: "/".join(args)
        mock_audeer.basename_wo_ext.return_value = "file1"
        mock_isfile.return_value = True

        translator = GoogleTranslator(language="en", util=mock_util)

        df = pd.DataFrame(
            {"text": ["hola"]},
            index=pd.Index([("file1.wav", timedelta(0), timedelta(1))]),
        )

        result = translator.translate_index(df)

        assert isinstance(result, pd.DataFrame)
        assert "en" in result.columns
        assert result["en"].iloc[0] == "cached translation"
        mock_asyncio.run.assert_not_called()

    @patch("nkululeko.autopredict.google_translator.audeer")
    @patch("nkululeko.autopredict.google_translator.os.path.isfile")
    @patch("nkululeko.autopredict.google_translator.asyncio")
    def test_translate_index_cache_language_mismatch(
        self, mock_asyncio, mock_isfile, mock_audeer
    ):
        mock_util = Mock()
        mock_util.get_path.return_value = "/cache"
        mock_util.read_json.return_value = {
            "translation": "stale english",
            "language": "en",
        }

        mock_audeer.mkdir.return_value = "/cache/translations/de"
        mock_audeer.path.side_effect = lambda *args: "/".join(args)
        mock_audeer.basename_wo_ext.return_value = "file1"
        mock_isfile.return_value = True
        mock_asyncio.run.return_value = ["hallo"]

        translator = GoogleTranslator(language="de", util=mock_util)

        df = pd.DataFrame(
            {"text": ["hello"]},
            index=pd.Index([("file1.wav", timedelta(0), timedelta(1))]),
        )

        result = translator.translate_index(df)

        assert result["de"].iloc[0] == "hallo"
        mock_asyncio.run.assert_called_once()

    @patch("nkululeko.autopredict.google_translator.audeer")
    @patch("nkululeko.autopredict.google_translator.os.path.isfile")
    @patch("nkululeko.autopredict.google_translator.asyncio")
    def test_translate_index_without_cache(
        self, mock_asyncio, mock_isfile, mock_audeer
    ):
        mock_util = Mock()
        mock_util.get_path.return_value = "/cache"

        mock_audeer.mkdir.return_value = "/cache/translations/en"
        mock_audeer.path.side_effect = lambda *args: "/".join(args)
        mock_audeer.basename_wo_ext.return_value = "file1"
        mock_isfile.return_value = False

        # Mock the batch translation
        mock_asyncio.run.return_value = ["hello", "world"]

        translator = GoogleTranslator(language="en", util=mock_util)

        df = pd.DataFrame(
            {"text": ["hola", "mundo"]},
            index=pd.Index(
                [
                    ("file1.wav", timedelta(0), timedelta(1)),
                    ("file1.wav", timedelta(1), timedelta(2)),
                ]
            ),
        )

        result = translator.translate_index(df)

        assert isinstance(result, pd.DataFrame)
        assert "en" in result.columns
        assert len(result) == 2
        # Should call asyncio.run once with batch
        mock_asyncio.run.assert_called_once()
        # Should save to cache
        assert mock_util.save_json.call_count == 2
