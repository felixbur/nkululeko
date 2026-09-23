from unittest.mock import MagicMock

import numpy as np
import pytest

from nkululeko.feat_extract.feats_emotion2vec_emotion import (
    Emotion2vec_emotion,
    _clean_label,
)


class TestCleanLabel:
    def test_plain_label_unchanged(self):
        assert _clean_label("angry") == "angry"

    def test_bilingual_label_takes_part_after_slash(self):
        assert _clean_label("生气/angry") == "angry"

    def test_strips_whitespace(self):
        assert _clean_label(" happy ") == "happy"


class TestScoresFromResult:
    def _make_extractor(self):
        ext = Emotion2vec_emotion.__new__(Emotion2vec_emotion)
        ext.util = MagicMock()
        return ext

    def test_parses_labels_and_scores(self):
        ext = self._make_extractor()
        res = [{"labels": ["angry", "happy", "neutral"], "scores": [0.1, 0.7, 0.2]}]
        scores = ext._scores_from_result(res, "some_file.wav")
        assert scores == {"angry": 0.1, "happy": 0.7, "neutral": 0.2}

    def test_parses_bilingual_labels(self):
        ext = self._make_extractor()
        res = [{"labels": ["生气/angry", "开心/happy"], "scores": [0.3, 0.6]}]
        scores = ext._scores_from_result(res, "some_file.wav")
        assert scores == {"angry": 0.3, "happy": 0.6}

    def test_empty_result_returns_empty_dict(self):
        ext = self._make_extractor()
        assert ext._scores_from_result([], "some_file.wav") == {}
        assert ext._scores_from_result(None, "some_file.wav") == {}


class TestPredictOne:
    def _make_extractor(self):
        ext = Emotion2vec_emotion.__new__(Emotion2vec_emotion)
        ext.util = MagicMock()
        ext.model = MagicMock()
        return ext

    def test_predict_one_uses_whole_file_when_no_segment(self):
        ext = self._make_extractor()
        ext.model.generate.return_value = [
            {"labels": ["angry", "happy"], "scores": [0.2, 0.8]}
        ]
        result = ext._predict_one("file.wav", None, None)
        assert result == {"angry": 0.2, "happy": 0.8}
        ext.model.generate.assert_called_once_with(
            "file.wav", granularity="utterance", extract_embedding=False
        )

    def test_predict_one_returns_empty_on_model_error(self):
        ext = self._make_extractor()
        ext.model.generate.side_effect = RuntimeError("boom")
        result = ext._predict_one("file.wav", None, None)
        assert result == {}
        ext.util.warn.assert_called_once()


class TestExtractSample:
    def test_extract_sample_returns_score_array(self, tmp_path, monkeypatch):
        ext = Emotion2vec_emotion.__new__(Emotion2vec_emotion)
        ext.util = MagicMock()
        ext.model_initialized = True
        ext.model = MagicMock()
        ext.model.generate.return_value = [
            {"labels": ["angry", "happy"], "scores": [0.3, 0.7]}
        ]
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        signal = np.zeros((1, 1600), dtype=np.float32)
        feats = ext.extract_sample(signal, 16000)

        np.testing.assert_allclose(feats, [0.3, 0.7])
        # temp file must be cleaned up
        assert list(tmp_path.glob("tmp*")) == []
