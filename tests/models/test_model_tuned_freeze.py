"""Unit tests for TunedModel._freeze_encoder_layers (nkululeko/models/model_tuned.py).

Exercises the method directly against lightweight stand-ins for the
wav2vec2 encoder layer structure, rather than instantiating TunedModel
(which downloads a real HuggingFace model).
"""

import types

import pytest
import torch

from nkululeko.models.model_tuned import TunedModel
from nkululeko.utils.errors import NkululukoError


class DummyUtil:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)

    def debug(self, message):
        pass

    def error(self, message):
        raise NkululukoError(message)


def make_dummy_model(num_layers):
    layers = [torch.nn.Linear(2, 2) for _ in range(num_layers)]
    encoder = types.SimpleNamespace(layers=layers)
    wav2vec2 = types.SimpleNamespace(encoder=encoder)
    return types.SimpleNamespace(wav2vec2=wav2vec2)


def all_require_grad(layers, expected):
    return all(
        param.requires_grad == expected for layer in layers for param in layer.parameters()
    )


class TestFreezeEncoderLayers:
    def test_zero_is_noop(self):
        util = DummyUtil()
        fake_self = types.SimpleNamespace(util=util)
        model = make_dummy_model(4)
        TunedModel._freeze_encoder_layers(fake_self, model, 0)
        assert all_require_grad(model.wav2vec2.encoder.layers, True)
        assert util.warnings == []

    def test_freezes_first_n_leaves_rest_trainable(self):
        util = DummyUtil()
        fake_self = types.SimpleNamespace(util=util)
        model = make_dummy_model(4)
        TunedModel._freeze_encoder_layers(fake_self, model, 2)
        layers = model.wav2vec2.encoder.layers
        assert all_require_grad(layers[:2], False)
        assert all_require_grad(layers[2:], True)

    def test_freeze_more_than_available_freezes_all_and_warns(self):
        util = DummyUtil()
        fake_self = types.SimpleNamespace(util=util)
        model = make_dummy_model(3)
        TunedModel._freeze_encoder_layers(fake_self, model, 10)
        assert all_require_grad(model.wav2vec2.encoder.layers, False)
        assert len(util.warnings) == 1


def make_fake_self(num_layers, freeze_layers):
    util = DummyUtil()
    cfg = types.SimpleNamespace(freeze_layers=freeze_layers)
    return types.SimpleNamespace(util=util, num_layers=num_layers, cfg=cfg)


class TestValidateLayerConfig:
    def test_defaults_pass(self):
        fake_self = make_fake_self(num_layers=None, freeze_layers=0)
        TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_num_layers_within_range_passes(self):
        fake_self = make_fake_self(num_layers=6, freeze_layers=2)
        TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_num_layers_equal_to_original_passes(self):
        fake_self = make_fake_self(num_layers=24, freeze_layers=0)
        TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_num_layers_exceeding_original_errors(self):
        fake_self = make_fake_self(num_layers=30, freeze_layers=0)
        with pytest.raises(NkululukoError):
            TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_num_layers_zero_errors(self):
        fake_self = make_fake_self(num_layers=0, freeze_layers=0)
        with pytest.raises(NkululukoError):
            TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_freeze_layers_equal_to_num_layers_errors(self):
        fake_self = make_fake_self(num_layers=6, freeze_layers=6)
        with pytest.raises(NkululukoError):
            TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_freeze_layers_exceeding_num_layers_errors(self):
        fake_self = make_fake_self(num_layers=6, freeze_layers=10)
        with pytest.raises(NkululukoError):
            TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_freeze_layers_exceeding_original_when_num_layers_unset_errors(self):
        fake_self = make_fake_self(num_layers=None, freeze_layers=24)
        with pytest.raises(NkululukoError):
            TunedModel._validate_layer_config(fake_self, original_num_layers=24)

    def test_freeze_layers_one_less_than_num_layers_passes(self):
        fake_self = make_fake_self(num_layers=6, freeze_layers=5)
        TunedModel._validate_layer_config(fake_self, original_num_layers=24)
