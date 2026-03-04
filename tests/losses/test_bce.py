"""Unit tests for BCE loss functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from nkululeko.losses.loss_bce import (
    BCEWithLogitsLoss,
    WeightedBCEWithLogitsLoss,
    compute_class_weights,
)


@pytest.fixture
def sample_inputs():
    """Sample logits for testing."""
    return torch.tensor([0.5, -0.3, 1.2, -0.8, 0.1])


@pytest.fixture
def sample_targets():
    """Sample binary targets for testing."""
    return torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])


@pytest.fixture
def sample_labels():
    """Sample label array for compute_class_weights."""
    return ["fake", "fake", "fake", "fake", "real", "real"]


class TestBCEWithLogitsLoss:
    """Tests for standard BCE loss."""

    def test_basic_functionality(self, sample_inputs, sample_targets):
        """Test basic BCE loss computation."""
        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(sample_inputs, sample_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.dim() == 0  # scalar output

    def test_reduction_mean(self, sample_inputs, sample_targets):
        """Test mean reduction (default)."""
        loss_fn = BCEWithLogitsLoss(reduction="mean")
        loss = loss_fn(sample_inputs, sample_targets)

        # Compare with PyTorch's native implementation
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            sample_inputs, sample_targets, reduction="mean"
        )

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_reduction_sum(self, sample_inputs, sample_targets):
        """Test sum reduction."""
        loss_fn = BCEWithLogitsLoss(reduction="sum")
        loss = loss_fn(sample_inputs, sample_targets)

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            sample_inputs, sample_targets, reduction="sum"
        )

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_reduction_none(self, sample_inputs, sample_targets):
        """Test no reduction (per-sample loss)."""
        loss_fn = BCEWithLogitsLoss(reduction="none")
        loss = loss_fn(sample_inputs, sample_targets)

        assert loss.shape == sample_inputs.shape
        assert (loss >= 0).all()

    def test_perfect_prediction(self):
        """Test loss with perfect predictions."""
        # Very high logits for positive class
        inputs = torch.tensor([10.0, 10.0, 10.0])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(inputs, targets)

        assert loss.item() < 0.01  # Should be very small

    def test_worst_prediction(self):
        """Test loss with worst predictions."""
        # Very negative logits for positive class
        inputs = torch.tensor([-10.0, -10.0, -10.0])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(inputs, targets)

        assert loss.item() > 10  # Should be very large

    def test_2d_input(self):
        """Test with 2D input (N, 1) shape."""
        inputs = torch.tensor([[0.5], [-0.3], [1.2]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])

        loss_fn = BCEWithLogitsLoss()
        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestWeightedBCEWithLogitsLoss:
    """Tests for weighted BCE loss."""

    def test_basic_functionality(self, sample_inputs, sample_targets):
        """Test basic weighted BCE loss computation."""
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=2.0)
        loss = loss_fn(sample_inputs, sample_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.dim() == 0

    def test_pos_weight_scalar(self, sample_inputs, sample_targets):
        """Test with scalar pos_weight."""
        pos_weight = 2.0
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(sample_inputs, sample_targets)

        # Compare with manual calculation
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            sample_inputs,
            sample_targets,
            pos_weight=torch.tensor([pos_weight]),
            reduction="mean",
        )

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_pos_weight_tensor(self, sample_inputs, sample_targets):
        """Test with tensor pos_weight."""
        pos_weight = torch.tensor([3.0])
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(sample_inputs, sample_targets)

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            sample_inputs, sample_targets, pos_weight=pos_weight, reduction="mean"
        )

        assert torch.allclose(loss, expected, atol=1e-6)

    def test_pos_weight_effect(self):
        """Test that higher pos_weight increases loss for false negatives."""
        inputs = torch.tensor([-1.0, -1.0])  # Predicting negative
        targets = torch.tensor([1.0, 1.0])  # Actually positive (false negative)

        loss_low = WeightedBCEWithLogitsLoss(pos_weight=1.0)(inputs, targets)
        loss_high = WeightedBCEWithLogitsLoss(pos_weight=5.0)(inputs, targets)

        # Higher pos_weight should increase loss for false negatives
        assert loss_high > loss_low

    def test_reduction_modes(self, sample_inputs, sample_targets):
        """Test different reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = WeightedBCEWithLogitsLoss(pos_weight=2.0, reduction=reduction)
            loss = loss_fn(sample_inputs, sample_targets)

            if reduction == "none":
                assert loss.shape == sample_inputs.shape
            else:
                assert loss.dim() == 0

    def test_device_compatibility(self, sample_inputs, sample_targets):
        """Test that pos_weight moves to correct device."""
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=2.0)

        # CPU test
        loss_cpu = loss_fn(sample_inputs, sample_targets)
        assert loss_cpu.device.type == "cpu"

    def test_no_weight_equals_standard_bce(self, sample_inputs, sample_targets):
        """Test that pos_weight=1.0 equals standard BCE."""
        weighted_loss = WeightedBCEWithLogitsLoss(pos_weight=1.0)(
            sample_inputs, sample_targets
        )
        standard_loss = BCEWithLogitsLoss()(sample_inputs, sample_targets)

        assert torch.allclose(weighted_loss, standard_loss, atol=1e-6)


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_balanced_mode_basic(self, sample_labels):
        """Test balanced mode with basic input."""
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(sample_labels, label_names, mode="balanced")

        # 4 fake, 2 real -> pos_weight = 4/2 = 2.0
        assert pos_weight == 2.0

    def test_balanced_mode_equal_distribution(self):
        """Test balanced mode with equal class distribution."""
        labels = ["fake", "fake", "real", "real"]
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 1.0

    def test_sqrt_mode(self, sample_labels):
        """Test sqrt mode."""
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(sample_labels, label_names, mode="sqrt")

        # 4 fake, 2 real -> pos_weight = sqrt(4/2) = sqrt(2) ≈ 1.414
        expected = np.sqrt(2.0)
        assert np.isclose(pos_weight, expected, atol=1e-6)

    def test_with_pandas_series(self):
        """Test with pandas Series input."""
        labels = pd.Series(["fake", "fake", "fake", "real"])
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 3.0

    def test_with_numpy_array(self):
        """Test with numpy array input."""
        labels = np.array(["fake", "fake", "real"])
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 2.0

    def test_zero_positive_samples(self):
        """Test edge case with no positive samples."""
        labels = ["fake", "fake", "fake"]
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        # Should return 1.0 to avoid division by zero
        assert pos_weight == 1.0

    def test_highly_imbalanced(self):
        """Test with highly imbalanced dataset."""
        labels = ["fake"] * 100 + ["real"] * 10
        label_names = ["fake", "real"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 10.0

    def test_invalid_mode(self, sample_labels):
        """Test that invalid mode raises ValueError."""
        label_names = ["fake", "real"]

        with pytest.raises(ValueError, match="Unknown mode"):
            compute_class_weights(sample_labels, label_names, mode="invalid")

    def test_different_label_names(self):
        """Test with different label names."""
        labels = ["negative", "negative", "positive"]
        label_names = ["negative", "positive"]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 2.0

    def test_numeric_labels(self):
        """Test with numeric labels (0s and 1s)."""
        labels = [0, 0, 0, 1]
        label_names = [0, 1]
        pos_weight = compute_class_weights(labels, label_names, mode="balanced")

        assert pos_weight == 3.0
