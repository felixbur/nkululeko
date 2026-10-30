"""Unit tests for MLP_Reg_model's inner MLP architecture
(nkululeko/models/model_mlp_regression.py)."""

import torch

from nkululeko.models.model_mlp_regression import MLP_Reg_model


class TestEveryHiddenLayerGetsAnActivation:
    """GH #467: with 2+ hidden layers, the activation key after the first
    later Linear layer ("0_r") collided with the one set before the loop,
    silently overwriting it -- one fewer activation than Linear layers,
    and the last hidden layer had none at all. Same bug and fix as
    model_mlp.py's MLP (test_model_mlp.py's
    TestEveryHiddenLayerGetsAnActivation), duplicated here since this is a
    separate (copy-pasted) MLP class."""

    def _activation_count(self, mlp_inner):
        return sum(
            1 for layer in mlp_inner.linear if isinstance(layer, torch.nn.ReLU)
        )

    def _linear_count(self, mlp_inner):
        return sum(
            1 for layer in mlp_inner.linear if isinstance(layer, torch.nn.Linear)
        )

    def test_two_hidden_layers_get_two_activations(self):
        mlp_inner = MLP_Reg_model.MLP(10, [16, 8], 1, False, torch.nn.ReLU())
        assert self._activation_count(mlp_inner) == self._linear_count(mlp_inner) - 1

    def test_three_hidden_layers_get_three_activations(self):
        mlp_inner = MLP_Reg_model.MLP(10, [32, 16, 8], 1, False, torch.nn.ReLU())
        assert self._activation_count(mlp_inner) == self._linear_count(mlp_inner) - 1

    def test_no_two_linear_layers_are_adjacent(self):
        mlp_inner = MLP_Reg_model.MLP(10, [32, 16, 8], 1, False, torch.nn.ReLU())
        modules = list(mlp_inner.linear)
        for prev, nxt in zip(modules, modules[1:]):
            assert not (
                isinstance(prev, torch.nn.Linear) and isinstance(nxt, torch.nn.Linear)
            )
