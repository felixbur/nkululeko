"""Unit tests for Model's backbone class selection (nkululeko/models/model_tuned.py).

Regression coverage for GitHub issue #150: finetuning a WavLM checkpoint
silently built a plain Wav2Vec2Model backbone regardless of the configured
pretrained_model, so WavLM-specific attention weights ("gated relative
position bias") never had a matching submodule to load into. These tests
build minimal configs directly (no network/download) and check that Model
picks the backbone class matching config.model_type.
"""

from transformers import HubertConfig, HubertModel, Wav2Vec2Config, WavLMConfig, WavLMModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from nkululeko.models.model_tuned import Model


def make_config(config_cls, **overrides):
    config = config_cls(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16, 16),
        conv_stride=(2, 2),
        conv_kernel=(3, 3),
        # Mirrors what _init_huggingface_model sets on the real config:
        # "eager" is universally supported, unlike the SDPA default (which
        # e.g. WavLM doesn't implement as of transformers 5).
        attn_implementation="eager",
    )
    config.final_dropout = 0.1
    config.is_classifier = True
    config.num_labels = 2
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestBackboneSelection:
    def test_wav2vec2_config_builds_wav2vec2_model(self):
        model = Model(make_config(Wav2Vec2Config))
        assert type(model.wav2vec2) is Wav2Vec2Model

    def test_wavlm_config_builds_wavlm_model(self):
        model = Model(make_config(WavLMConfig))
        assert type(model.wav2vec2) is WavLMModel

    def test_hubert_config_builds_hubert_model(self):
        model = Model(make_config(HubertConfig))
        assert type(model.wav2vec2) is HubertModel

    def test_unrecognized_model_type_falls_back_to_wav2vec2(self):
        config = make_config(Wav2Vec2Config, model_type="some-future-architecture")
        model = Model(config)
        assert type(model.wav2vec2) is Wav2Vec2Model
