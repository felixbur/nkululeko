"""Unit tests for Model.forward()'s eval-mode softmax (nkululeko/models/model_tuned.py).

Regression: `if not self.training: logits = torch.softmax(logits, dim=1)` applied
softmax unconditionally during eval, regardless of is_classifier. Regression
models have num_labels==1, so softmax over that single-element dimension
always returns exactly 1.0 no matter the input - silently turning every
eval/test regression prediction into the same constant. This made finetuned
regression models look completely untrained (frozen, input-independent
eval/test metrics) even after real training moved the loss significantly.
Classification softmax must be unaffected, so it's tested here too.
"""

import torch

from nkululeko.models.model_tuned import Model
from tests.models.test_model_tuned_backbone import make_config
from transformers import Wav2Vec2Config


class TestEvalModeSoftmax:
    def test_regression_eval_output_is_not_collapsed_to_constant(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        model.eval()

        input_a = torch.randn(1, 400)
        input_b = torch.randn(1, 400) * 5 + 3

        with torch.no_grad():
            logits_a = model(input_a).logits
            logits_b = model(input_b).logits

        assert logits_a.item() != 1.0
        assert logits_b.item() != 1.0
        assert not torch.isclose(logits_a, logits_b)

    def test_classification_eval_output_is_still_softmaxed(self):
        config = make_config(Wav2Vec2Config, is_classifier=True, num_labels=3)
        model = Model(config)
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(2, 400)).logits

        assert torch.allclose(logits.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_regression_train_mode_output_unaffected(self):
        config = make_config(Wav2Vec2Config, is_classifier=False, num_labels=1)
        model = Model(config)
        model.train()

        logits = model(torch.randn(1, 400)).logits

        assert logits.item() != 1.0
