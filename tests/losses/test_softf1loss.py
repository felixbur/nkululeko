"""Unit tests for SoftF1Loss (nkululeko/losses/loss_softf1loss.py)."""

import pytest
import torch

from nkululeko.losses.loss_softf1loss import SoftF1Loss


class TestSoftF1LossBasic:
    def test_output_is_scalar(self):
        loss_fn = SoftF1Loss(num_classes=3)
        y_pred = torch.randn(4, 3)
        y_true = torch.tensor([0, 1, 2, 1])
        loss = loss_fn(y_pred, y_true)
        assert loss.dim() == 0

    def test_loss_in_zero_one_range(self):
        loss_fn = SoftF1Loss(num_classes=3)
        y_pred = torch.randn(8, 3)
        y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        loss = loss_fn(y_pred, y_true)
        assert 0.0 <= loss.item() <= 1.0

    def test_returns_tensor(self):
        loss_fn = SoftF1Loss(num_classes=2)
        loss = loss_fn(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        assert isinstance(loss, torch.Tensor)


class TestSoftF1LossPerfectPredictions:
    def test_perfect_predictions_yield_near_zero_loss(self):
        num_classes = 3
        loss_fn = SoftF1Loss(num_classes=num_classes)
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.zeros(3, num_classes)
        y_pred[0, 0] = 10.0
        y_pred[1, 1] = 10.0
        y_pred[2, 2] = 10.0
        loss = loss_fn(y_pred, y_true)
        assert loss.item() < 0.01

    def test_random_predictions_higher_than_perfect(self):
        num_classes = 3
        loss_fn = SoftF1Loss(num_classes=num_classes)
        y_true = torch.tensor([0, 1, 2])

        y_pred_perfect = torch.zeros(3, num_classes)
        y_pred_perfect[0, 0] = 10.0
        y_pred_perfect[1, 1] = 10.0
        y_pred_perfect[2, 2] = 10.0

        torch.manual_seed(42)
        y_pred_random = torch.randn(3, num_classes)

        loss_perfect = loss_fn(y_pred_perfect, y_true)
        loss_random = loss_fn(y_pred_random, y_true)
        assert loss_perfect.item() < loss_random.item()


class TestSoftF1LossAssertions:
    def test_raises_on_1d_pred(self):
        loss_fn = SoftF1Loss(num_classes=3)
        with pytest.raises(AssertionError):
            loss_fn(torch.randn(4), torch.tensor([0, 1, 2, 1]))

    def test_raises_on_2d_true(self):
        loss_fn = SoftF1Loss(num_classes=3)
        with pytest.raises(AssertionError):
            loss_fn(torch.randn(4, 3), torch.zeros(4, 1, dtype=torch.long))


class TestSoftF1LossWeighted:
    def test_weighted_loss_is_scalar_in_range(self):
        num_classes = 2
        weight = torch.tensor([1.0, 2.0])
        loss_fn = SoftF1Loss(weight=weight, num_classes=num_classes)
        y_pred = torch.randn(4, num_classes)
        y_true = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(y_pred, y_true)
        assert 0.0 <= loss.item() <= 1.0

    def test_weighted_differs_from_unweighted(self):
        torch.manual_seed(0)
        num_classes = 2
        y_pred = torch.randn(6, num_classes)
        y_true = torch.tensor([0, 1, 0, 1, 0, 1])
        unweighted = SoftF1Loss(num_classes=num_classes)(y_pred, y_true)
        weighted = SoftF1Loss(
            weight=torch.tensor([1.0, 10.0]), num_classes=num_classes
        )(y_pred, y_true)
        assert unweighted.item() != pytest.approx(weighted.item())


class TestSoftF1LossEpsilon:
    def test_custom_epsilon(self):
        loss_fn = SoftF1Loss(epsilon=1e-4, num_classes=2)
        loss = loss_fn(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        assert 0.0 <= loss.item() <= 1.0
