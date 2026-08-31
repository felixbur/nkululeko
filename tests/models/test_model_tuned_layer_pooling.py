"""Unit tests for Model's weighted layer_pooling (nkululeko/models/model_tuned.py).

Regression coverage for [FINETUNE] layer_pooling: "weighted" adds one
learnable scalar per encoder layer (softmax-normalized) and pools a weighted
sum of every layer's hidden states instead of only the last one - the
SUPERB-benchmark "weighted sum of hidden states" technique. Useful
paralinguistic/prosodic signal is often concentrated in earlier or middle
layers, not necessarily the last one.
"""

import torch
from transformers import Wav2Vec2Config

from nkululeko.models.model_tuned import Model
from tests.models.test_model_tuned_backbone import make_config


class TestLayerPoolingDefault:
    def test_default_is_last_layer_only(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        assert model.layer_pooling_mode == "last"
        assert not hasattr(model, "layer_weights")

    def test_default_forward_matches_last_hidden_state_path(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 400))
        assert out.logits.shape == (1, 1)


class TestLayerPoolingWeighted:
    def test_creates_one_weight_per_layer_plus_input(self):
        config = make_config(
            Wav2Vec2Config,
            is_classifier=False,
            num_labels=1,
            layer_pooling="weighted",
        )
        model = Model(config)
        assert model.layer_pooling_mode == "weighted"
        # +1 for the encoder's input hidden states (index 0 of
        # output_hidden_states), matching config.num_hidden_layers layers.
        assert model.layer_weights.shape == (config.num_hidden_layers + 1,)

    def test_layer_weights_are_trainable(self):
        config = make_config(
            Wav2Vec2Config,
            is_classifier=False,
            num_labels=1,
            layer_pooling="weighted",
        )
        model = Model(config)
        assert model.layer_weights.requires_grad

    def test_zero_init_gives_uniform_softmax_weights(self):
        config = make_config(
            Wav2Vec2Config,
            is_classifier=False,
            num_labels=1,
            layer_pooling="weighted",
        )
        model = Model(config)
        weights = torch.softmax(model.layer_weights, dim=0)
        expected = 1.0 / (config.num_hidden_layers + 1)
        assert torch.allclose(weights, torch.full_like(weights, expected), atol=1e-6)

    def test_forward_produces_correct_output_shape(self):
        config = make_config(
            Wav2Vec2Config,
            is_classifier=False,
            num_labels=1,
            layer_pooling="weighted",
        )
        model = Model(config)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 400))
        assert out.logits.shape == (1, 1)

    def test_changing_layer_weights_changes_pooled_output(self):
        # Sanity check the weighted sum actually depends on all layers, not
        # just collapsing to one regardless of weight values.
        config = make_config(
            Wav2Vec2Config,
            is_classifier=False,
            num_labels=1,
            layer_pooling="weighted",
        )
        model = Model(config)
        model.eval()
        signal = torch.randn(1, 400)

        with torch.no_grad():
            out_before = model(signal).logits.clone()
            model.layer_weights.data = torch.randn_like(model.layer_weights)
            out_after = model(signal).logits

        assert not torch.allclose(out_before, out_after)

class TestWeightedLayerSum:
    """Unit tests for Model._weighted_layer_sum in isolation - synthetic,
    clearly-distinguishable layers rather than a real transformer forward
    pass, since a tiny randomly-initialized test transformer's layers can
    end up numerically close enough to make gradient-nonzero checks flaky.
    """

    def test_uniform_weights_average_the_layers(self):
        layers = (torch.zeros(1, 2, 3), torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0))
        weights = torch.zeros(3)  # softmax -> uniform

        result = Model._weighted_layer_sum(layers, weights)

        assert torch.allclose(result, torch.full((1, 2, 3), 1.0))

    def test_dominant_weight_selects_that_layer(self):
        layers = (torch.zeros(1, 2, 3), torch.ones(1, 2, 3), torch.full((1, 2, 3), 5.0))
        weights = torch.tensor([-100.0, -100.0, 100.0])  # softmax -> ~[0, 0, 1]

        result = Model._weighted_layer_sum(layers, weights)

        assert torch.allclose(result, torch.full((1, 2, 3), 5.0), atol=1e-4)

    def test_gradient_flows_to_layer_weights(self):
        layers = (torch.zeros(1, 2, 3), torch.ones(1, 2, 3), torch.full((1, 2, 3), 5.0))
        weights = torch.zeros(3, requires_grad=True)

        result = Model._weighted_layer_sum(layers, weights)
        result.sum().backward()

        assert weights.grad is not None
        assert not torch.all(weights.grad == 0)
