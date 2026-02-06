from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd

from nkululeko.autopredict.ap_text import TextPredictor


class TestTextPredictor:
    @patch("nkululeko.autopredict.ap_text.torch")
    @patch("nkululeko.autopredict.whisper_transcriber.Transcriber")
    def test_init_with_util(self, mock_transcriber, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_util = Mock()
        mock_util.config_val.side_effect = lambda section, key, default: default
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = TextPredictor(df, util=mock_util)
        
        assert predictor.df is df
        assert predictor.util is mock_util
        mock_transcriber.assert_called_once()

    @patch("nkululeko.autopredict.ap_text.torch")
    @patch("nkululeko.autopredict.whisper_transcriber.Transcriber")
    @patch("nkululeko.autopredict.ap_text.Util")
    def test_init_without_util(self, mock_util_class, mock_transcriber, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_util = Mock()
        mock_util.config_val.side_effect = lambda section, key, default: default
        mock_util_class.return_value = mock_util
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        predictor = TextPredictor(df)
        
        assert predictor.df is df
        mock_util_class.assert_called_once_with("textPredictor")

    @patch("nkululeko.autopredict.ap_text.torch")
    @patch("nkululeko.autopredict.whisper_transcriber.Transcriber")
    def test_predict(self, mock_transcriber_class, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_util = Mock()
        mock_util.config_val.side_effect = lambda section, key, default: default
        
        # Create mock transcriber
        mock_transcriber = Mock()
        mock_text_df = pd.DataFrame(
            {"text": ["hello", "world", "test"]},
            index=pd.Index([
                ("file1.wav", timedelta(0), timedelta(1)),
                ("file2.wav", timedelta(0), timedelta(1)),
                ("file3.wav", timedelta(0), timedelta(1))
            ])
        )
        mock_transcriber.transcribe_index.return_value = mock_text_df
        mock_transcriber_class.return_value = mock_transcriber
        
        df = pd.DataFrame(
            {"dummy": [1, 2, 3]},
            index=pd.Index([
                ("file1.wav", timedelta(0), timedelta(1)),
                ("file2.wav", timedelta(0), timedelta(1)),
                ("file3.wav", timedelta(0), timedelta(1))
            ])
        )
        predictor = TextPredictor(df, util=mock_util)
        
        result = predictor.predict("train")
        
        assert "text" in result.columns
        assert len(result) == 3
        assert list(result["text"]) == ["hello", "world", "test"]
        mock_transcriber.transcribe_index.assert_called_once()
