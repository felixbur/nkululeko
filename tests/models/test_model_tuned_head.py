"""Unit tests for ModelHead's configurable architecture (model_tuned.py).

ModelHead used to hardcode a single hidden layer sized to the backbone's own
hidden_size, with a fixed tanh activation - so finetuning's head architecture
could never be matched to mlp/mlp_reg's [MODEL] layers/activation, making any
embeddings-vs-finetuning comparison confounded by two different head shapes
on top of the same backbone. head_layers/head_activation (from
[FINETUNE] head_layers/head_activation) make both configurable while keeping
the original architecture as the default.
"""

import types

import torch

from nkululeko.models.model_tuned import ModelHead


def make_config(hidden_size=8, num_labels=1, final_dropout=0.0, **overrides):
    config = types.SimpleNamespace(
        hidden_size=hidden_size,
        num_labels=num_labels,
        final_dropout=final_dropout,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestModelHeadDefaults:
    def test_default_is_single_hidden_layer_sized_to_backbone(self):
        config = make_config(hidden_size=8, num_labels=1)
        head = ModelHead(config)

        linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 2
        assert linears[0].in_features == 8 and linears[0].out_features == 8
        assert linears[1].in_features == 8 and linears[1].out_features == 1

    def test_default_activation_is_tanh(self):
        config = make_config()
        head = ModelHead(config)
        assert any(isinstance(m, torch.nn.Tanh) for m in head.net)

    def test_forward_produces_correct_output_shape(self):
        config = make_config(hidden_size=8, num_labels=3)
        head = ModelHead(config)
        out = head(torch.randn(4, 8))
        assert out.shape == (4, 3)


class TestModelHeadConfigurable:
    def test_custom_head_layers_builds_matching_linear_stack(self):
        config = make_config(hidden_size=8, num_labels=1, head_layers=[1024, 256])
        head = ModelHead(config)

        linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in linears] == [
            (8, 1024),
            (1024, 256),
            (256, 1),
        ]

    def test_custom_activation_relu(self):
        config = make_config(head_activation="relu")
        head = ModelHead(config)
        assert any(isinstance(m, torch.nn.ReLU) for m in head.net)
        assert not any(isinstance(m, torch.nn.Tanh) for m in head.net)

    def test_unknown_activation_raises(self):
        config = make_config(head_activation="not-a-real-activation")
        try:
            ModelHead(config)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "not-a-real-activation" in str(e)

    def test_none_head_layers_falls_back_to_hidden_size(self):
        config = make_config(hidden_size=8, num_labels=1, head_layers=None)
        head = ModelHead(config)
        linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
        assert [(l.in_features, l.out_features) for l in linears] == [(8, 8), (8, 1)]
