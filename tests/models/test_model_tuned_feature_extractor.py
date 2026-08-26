"""Unit tests for TunedModel._build_feature_extractor (nkululeko/models/model_tuned.py).

Regression coverage for GitHub issue #150: the feature extractor was
hand-built with do_normalize hardcoded True for every pretrained_model, but
some checkpoints (e.g. microsoft/wavlm-base) actually expect do_normalize
False - feeding them wrongly-scaled input. These tests mock
Wav2Vec2FeatureExtractor.from_pretrained to avoid any network access.
"""

import types
from unittest.mock import patch

import transformers

from nkululeko.models.model_tuned import TunedModel


class DummyUtil:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


def build(pretrained_model):
    fake_self = types.SimpleNamespace(util=DummyUtil())
    return TunedModel._build_feature_extractor(fake_self, pretrained_model), fake_self.util


class TestBuildFeatureExtractor:
    def test_uses_checkpoint_own_config_when_available(self):
        stub = transformers.Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, do_normalize=False
        )
        with patch.object(
            transformers.Wav2Vec2FeatureExtractor, "from_pretrained", return_value=stub
        ):
            extractor, util = build("microsoft/wavlm-base")
        assert extractor.do_normalize is False
        assert util.warnings == []

    def test_return_attention_mask_forced_true_even_if_checkpoint_says_false(self):
        stub = transformers.Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, return_attention_mask=False
        )
        with patch.object(
            transformers.Wav2Vec2FeatureExtractor, "from_pretrained", return_value=stub
        ):
            extractor, _ = build("some/checkpoint")
        assert extractor.return_attention_mask is True

    def test_falls_back_and_warns_when_checkpoint_has_no_preprocessor_config(self):
        with patch.object(
            transformers.Wav2Vec2FeatureExtractor,
            "from_pretrained",
            side_effect=OSError("no preprocessor_config.json"),
        ):
            extractor, util = build("local/custom-checkpoint")
        assert extractor.do_normalize is True
        assert extractor.sampling_rate == 16000
        assert extractor.return_attention_mask is True
        assert len(util.warnings) == 1
