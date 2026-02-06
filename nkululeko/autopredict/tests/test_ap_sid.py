from datetime import timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytest.importorskip("pyannote.audio")

from nkululeko.autopredict.ap_sid import SIDPredictor


class TestSIDPredictor:
    @patch("nkululeko.autopredict.ap_sid.Pipeline")
    @patch("nkululeko.autopredict.ap_sid.torch")
    def test_init_with_token(self, mock_torch, mock_pipeline_class):
        mock_util = Mock()
        mock_util.config_val.side_effect = lambda section, key, default: "test_token" if key == "hf_token" else "cpu"
        
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        
        with patch("nkululeko.autopredict.ap_sid.Util", return_value=mock_util):
            predictor = SIDPredictor(df)
        
        assert predictor.df is df
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch("nkululeko.autopredict.ap_sid.Pipeline")
    @patch("nkululeko.autopredict.ap_sid.torch")
    def test_init_without_token_raises_error(self, mock_torch, mock_pipeline_class):
        mock_util = Mock()
        mock_util.config_val.return_value = None
        mock_util.error.side_effect = Exception("Token required")
        
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        
        with patch("nkululeko.autopredict.ap_sid.Util", return_value=mock_util):
            with pytest.raises(Exception, match="Token required"):
                predictor = SIDPredictor(df)

    @patch("nkululeko.autopredict.ap_sid.concat_files")
    @patch("nkululeko.autopredict.ap_sid.Pipeline")
    @patch("nkululeko.autopredict.ap_sid.torch")
    def test_predict(self, mock_torch, mock_pipeline_class, mock_concat):
        mock_util = Mock()
        mock_util.config_val.side_effect = lambda section, key, default: "test_token" if key == "hf_token" else "cpu"
        mock_util.exist_pickle.return_value = False
        
        # Create mock annotation
        mock_annotation = Mock()
        mock_turn = Mock()
        mock_turn.start = 0.0
        mock_turn.end = 1.0
        mock_annotation.itertracks.return_value = [
            (mock_turn, None, "SPEAKER_01"),
            (mock_turn, None, "SPEAKER_01"),
            (mock_turn, None, "SPEAKER_02")
        ]
        
        mock_pipeline = Mock()
        mock_pipeline.return_value = mock_annotation
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        index = pd.Index([
            ("file1.wav", timedelta(0), timedelta(1)),
            ("file2.wav", timedelta(0), timedelta(1)),
            ("file3.wav", timedelta(0), timedelta(1))
        ])
        df = pd.DataFrame({"dummy": [1, 2, 3]}, index=index)
        
        with patch("nkululeko.autopredict.ap_sid.Util", return_value=mock_util):
            predictor = SIDPredictor(df)
            result = predictor.predict("train")
        
        assert "speaker" in result.columns
        assert len(result) == 3
        mock_concat.assert_called_once()
