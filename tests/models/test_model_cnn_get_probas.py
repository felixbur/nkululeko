"""Unit tests for CNNModel.get_probas() (nkululeko/models/model_cnn.py).

Regression (GH #446): identical bug to the one already fixed in the
sibling model_mlp.py - get_probas() wrote raw logits into the per-class
"probability" columns with no softmax. Logits aren't themselves
probabilities (not bounded to [0, 1], don't sum to 1), so the saved
columns - and anything derived from them (uncertainty, session averaging,
calibration, a probability threshold on ROC) - were silently wrong. argmax
(the predicted label) is unaffected either way, since softmax is
monotonic.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from nkululeko.models.model_cnn import CNNModel


@pytest.fixture
def cnn_model():
    model = CNNModel.__new__(CNNModel)
    model.df_test = pd.DataFrame({"label": [1, 0]})
    model.target = "label"
    return model


class TestGetProbasAppliesSoftmax:
    def test_probability_columns_are_valid_distributions(self, cnn_model):
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])

        probas = cnn_model.get_probas(logits)

        row_sums = probas.sum(axis=1).to_numpy()
        np.testing.assert_allclose(row_sums, np.ones(2), rtol=1e-5)
        assert (probas.to_numpy() >= 0).all()
        assert (probas.to_numpy() <= 1).all()

    def test_matches_manual_softmax(self, cnn_model):
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])
        expected = torch.softmax(logits, dim=1).numpy()

        probas = cnn_model.get_probas(logits)

        np.testing.assert_allclose(probas[0].to_numpy(), expected[:, 0], rtol=1e-5)
        np.testing.assert_allclose(probas[1].to_numpy(), expected[:, 1], rtol=1e-5)

    def test_argmax_label_unaffected(self, cnn_model):
        logits = torch.tensor([[-2.0, 3.0], [1.5, -0.5]])

        probas = cnn_model.get_probas(logits)

        expected_argmax = logits.argmax(dim=1).numpy()
        actual_argmax = probas.to_numpy().argmax(axis=1)
        np.testing.assert_array_equal(actual_argmax, expected_argmax)
