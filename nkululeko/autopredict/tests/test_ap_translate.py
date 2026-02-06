from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from nkululeko.autopredict.ap_translate import TextTranslator


class TestTextTranslator:
    @patch("nkululeko.autopredict.google_translator.GoogleTranslator")
    def test_init_with_util(self, mock_translator):
        mock_util = Mock()
        mock_util.config_val.return_value = "en"
        
        df = pd.DataFrame({"text": ["hola", "mundo"]})
        translator = TextTranslator(df, util=mock_util)
        
        assert translator.df is df
        assert translator.util is mock_util
        assert translator.language == "en"
        mock_translator.assert_called_once()

    @patch("nkululeko.autopredict.google_translator.GoogleTranslator")
    @patch("nkululeko.autopredict.ap_translate.Util")
    def test_init_without_util(self, mock_util_class, mock_translator):
        mock_util = Mock()
        mock_util.config_val.return_value = "en"
        mock_util_class.return_value = mock_util
        
        df = pd.DataFrame({"text": ["hola", "mundo"]})
        translator = TextTranslator(df)
        
        assert translator.df is df
        mock_util_class.assert_called_once_with("translator")

    @patch("nkululeko.autopredict.google_translator.GoogleTranslator")
    def test_predict(self, mock_translator_class):
        mock_util = Mock()
        mock_util.config_val.return_value = "en"
        
        # Create mock translator
        mock_translator = Mock()
        mock_translated_df = pd.DataFrame(
            {"en": ["hello", "world"]},
            index=pd.Index([
                ("file1.wav", timedelta(0), timedelta(1)),
                ("file2.wav", timedelta(0), timedelta(1))
            ])
        )
        mock_translator.translate_index.return_value = mock_translated_df
        mock_translator_class.return_value = mock_translator
        
        df = pd.DataFrame(
            {"text": ["hola", "mundo"]},
            index=pd.Index([
                ("file1.wav", timedelta(0), timedelta(1)),
                ("file2.wav", timedelta(0), timedelta(1))
            ])
        )
        translator = TextTranslator(df, util=mock_util)
        
        result = translator.predict("train")
        
        assert "en" in result.columns
        assert len(result) == 2
        assert list(result["en"]) == ["hello", "world"]
        mock_translator.translate_index.assert_called_once()
