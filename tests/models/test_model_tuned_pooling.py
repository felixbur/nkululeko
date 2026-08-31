"""Unit tests for Model.pooling()'s meanvar mode (nkululeko/models/model_tuned.py).

Regression coverage for the [FINETUNE] pooling config: "meanvar" concatenates
mean and variance along the feature dimension, doubling what the head
receives - if the head's input dim isn't updated to match, the first Linear
layer silently receives the wrong shape. Both the batched (attention_mask
present) and single-sample (attention_mask=None) pooling branches need their
own variance computation, matching the mean computation's own masking.
"""

import torch
from transformers import Wav2Vec2Config

from nkululeko.models.model_tuned import Model
from tests.models.test_model_tuned_backbone import make_config


def _raw_audio_length_for(model, target_feature_len):
    """Find a raw-audio attention_mask length that downsamples to target_feature_len.

    _get_feature_vector_attention_mask() maps a raw-audio-length mask through
    the conv feature extractor's own stride/kernel to a shorter framewise
    mask - it must match hidden_states' actual seq_len exactly or indexing
    inside it goes out of bounds. Deriving this from the model's own
    _get_feat_extract_output_lengths() (rather than hand-computing conv
    arithmetic) keeps this test correct regardless of make_config's own
    conv_stride/conv_kernel choices.
    """
    raw_len = target_feature_len
    while (
        model._get_feat_extract_output_lengths(torch.tensor([raw_len])).item()
        < target_feature_len
    ):
        raw_len += 1
    return raw_len


class TestPoolingMean:
    def test_default_pooling_output_dim_matches_hidden_size(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        hidden_states = torch.randn(2, 10, config.hidden_size)
        attention_mask = torch.ones(2, _raw_audio_length_for(model, 10))

        pooled = model.pooling(hidden_states, attention_mask)

        assert pooled.shape == (2, config.hidden_size)

    def test_batch_size_one_no_mask_pooling(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        hidden_states = torch.randn(1, 10, config.hidden_size)

        pooled = model.pooling(hidden_states, attention_mask=None)

        assert pooled.shape == (1, config.hidden_size)
        assert torch.allclose(pooled, hidden_states.mean(dim=1))


class TestPoolingMeanVar:
    def test_meanvar_doubles_output_dim(self):
        config = make_config(
            Wav2Vec2Config, is_classifier=False, num_labels=1, pooling="meanvar"
        )
        model = Model(config)
        hidden_states = torch.randn(2, 10, config.hidden_size)
        attention_mask = torch.ones(2, _raw_audio_length_for(model, 10))

        pooled = model.pooling(hidden_states, attention_mask)

        assert pooled.shape == (2, config.hidden_size * 2)

    def test_meanvar_no_mask_matches_torch_mean_and_var(self):
        config = make_config(
            Wav2Vec2Config, is_classifier=False, num_labels=1, pooling="meanvar"
        )
        model = Model(config)
        hidden_states = torch.randn(1, 10, config.hidden_size)

        pooled = model.pooling(hidden_states, attention_mask=None)

        expected_mean = hidden_states.mean(dim=1)
        expected_var = hidden_states.var(dim=1, unbiased=False)
        assert torch.allclose(pooled[:, : config.hidden_size], expected_mean, atol=1e-5)
        assert torch.allclose(pooled[:, config.hidden_size :], expected_var, atol=1e-5)

    def test_meanvar_variance_is_never_negative(self):
        # Regression guard: E[x^2] - E[x]^2 can go slightly negative from
        # floating-point cancellation even though true variance is >= 0.
        config = make_config(
            Wav2Vec2Config, is_classifier=False, num_labels=1, pooling="meanvar"
        )
        model = Model(config)
        # Near-constant hidden states: variance should be ~0, not negative.
        hidden_states = torch.full((2, 10, config.hidden_size), 3.0)
        attention_mask = torch.ones(2, _raw_audio_length_for(model, 10))

        pooled = model.pooling(hidden_states, attention_mask)

        variance_part = pooled[:, config.hidden_size :]
        assert (variance_part >= 0).all()

    def test_head_input_dim_matches_meanvar_pooling_output(self):
        config = make_config(
            Wav2Vec2Config, is_classifier=False, num_labels=1, pooling="meanvar"
        )
        model = Model(config)

        first_linear = next(m for m in model.head.net if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == config.hidden_size * 2
