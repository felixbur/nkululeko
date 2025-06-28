import os
import tempfile
from datetime import timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from nkululeko.autopredict.whisper_transcriber import Transcriber


class TestTranscriber:
    
    @patch('nkululeko.autopredict.whisper_transcriber.whisper.load_model')
    @patch('nkululeko.autopredict.whisper_transcriber.torch.cuda.is_available')
    def test_init_default_device(self, mock_cuda, mock_load_model):
        mock_cuda.return_value = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        transcriber = Transcriber()
        
        mock_load_model.assert_called_once_with("turbo", device="cuda")
        assert transcriber.language == "en"
        assert transcriber.model == mock_model

    @patch('nkululeko.autopredict.whisper_transcriber.whisper.load_model')
    def test_init_custom_params(self, mock_load_model):
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_util = Mock()
        
        transcriber = Transcriber(model_name="base", device="cpu", language="es", util=mock_util)
        
        mock_load_model.assert_called_once_with("base", device="cpu")
        assert transcriber.language == "es"
        assert transcriber.util == mock_util

    def test_transcribe_file(self):
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "  Hello world  "}
        
        transcriber = Transcriber()
        transcriber.model = mock_model
        
        result = transcriber.transcribe_file("test.wav")
        
        mock_model.transcribe.assert_called_once_with("test.wav", language="en", without_timestamps=True)
        assert result == "Hello world"

    @patch('nkululeko.autopredict.whisper_transcriber.audiofile.write')
    def test_transcribe_array(self, mock_write):
        transcriber = Transcriber()
        transcriber.transcribe_file = Mock(return_value="transcribed text")
        
        signal = np.array([0.1, 0.2, 0.3])
        sampling_rate = 16000
        
        result = transcriber.transcribe_array(signal, sampling_rate)
        
        mock_write.assert_called_once_with("temp.wav", signal, sampling_rate, format="wav")
        transcriber.transcribe_file.assert_called_once_with("temp.wav")
        assert result == "transcribed text"

    @patch('nkululeko.autopredict.whisper_transcriber.audiofile.read')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.mkdir')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.path')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.basename_wo_ext')
    @patch('nkululeko.autopredict.whisper_transcriber.os.path.isfile')
    def test_transcribe_index_with_cache(self, mock_isfile, mock_basename, mock_path, mock_mkdir, mock_read):
        mock_util = Mock()
        mock_util.get_path.return_value = "/cache"
        mock_util.read_json.return_value = {"transcription": "cached text"}
        
        mock_mkdir.return_value = "/cache/transcriptions"
        mock_path.side_effect = lambda *args: "/".join(args)
        mock_basename.return_value = "file1"
        mock_isfile.return_value = True
        
        transcriber = Transcriber(util=mock_util)
        
        index = pd.Index([
            ("file1.wav", timedelta(seconds=0), timedelta(seconds=1))
        ])
        
        result = transcriber.transcribe_index(index)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["text"] == "cached text"

    @patch('nkululeko.autopredict.whisper_transcriber.whisper.load_model')
    @patch('nkululeko.autopredict.whisper_transcriber.audiofile.read')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.mkdir')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.path')
    @patch('nkululeko.autopredict.whisper_transcriber.audeer.basename_wo_ext')
    @patch('nkululeko.autopredict.whisper_transcriber.os.path.isfile')
    def test_transcribe_index_without_cache(self, mock_isfile, mock_basename, mock_path, mock_mkdir, mock_audioread, mock_load_model):
        mock_util = Mock()
        mock_util.get_path.return_value = "/cache"
        
        mock_mkdir.return_value = "/cache/transcriptions"
        mock_path.side_effect = lambda *args: "/".join(args)
        mock_basename.return_value = "file1"
        mock_isfile.return_value = False
        mock_audioread.return_value = (np.array([0.1, 0.2]), 16000)
        mock_load_model.return_value = Mock()
        
        transcriber = Transcriber(util=mock_util)
        transcriber.transcribe_array = Mock(return_value="new transcription")
        
        index = pd.Index([
            ("file1.wav", timedelta(seconds=0), timedelta(seconds=1))
        ])
        
        result = transcriber.transcribe_index(index)
        
        mock_util.save_json.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["text"] == "new transcription"