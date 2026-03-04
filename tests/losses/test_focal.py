"""Unit tests for Focal Loss."""

import pytest
import torch
import torch.nn.functional as F

from nkululeko.losses.loss_focal import FocalLoss


@pytest.fixture
def sample_inputs():
    """Sample logits for testing."""
    return torch.tensor([0.5, -0.3, 1.2, -0.8, 0.1])


@pytest.fixture
def sample_targets():
    """Sample binary targets for testing."""
    return torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])


class TestFocalLoss:
    """Tests for Focal Loss function."""

    def test_basic_functionality(self, sample_inputs, sample_targets):
        """Test basic focal loss computation."""
        loss_fn = FocalLoss()
        loss = loss_fn(sample_inputs, sample_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.dim() == 0  # scalar output

    def test_default_parameters(self, sample_inputs, sample_targets):
        """Test default parameter values."""
        loss_fn = FocalLoss()

        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == "mean"

    def test_gamma_zero_equals_bce(self, sample_inputs, sample_targets):
        """Test that gamma=0 approximates BCE (with alpha weighting)."""
        focal_loss_fn = FocalLoss(alpha=None, gamma=0.0)
        focal_loss = focal_loss_fn(sample_inputs, sample_targets)

        bce_loss = F.binary_cross_entropy_with_logits(
            sample_inputs, sample_targets, reduction="mean"
        )

        # With gamma=0 and alpha=None, should equal BCE
        assert torch.allclose(focal_loss, bce_loss, atol=1e-5)

    def test_gamma_effect(self, sample_inputs, sample_targets):
        """Test that higher gamma focuses more on hard examples."""
        loss_gamma_0 = FocalLoss(alpha=None, gamma=0.0)(sample_inputs, sample_targets)
        loss_gamma_2 = FocalLoss(alpha=None, gamma=2.0)(sample_inputs, sample_targets)
        loss_gamma_5 = FocalLoss(alpha=None, gamma=5.0)(sample_inputs, sample_targets)

        # Higher gamma should reduce overall loss (down-weights easy examples)
        assert loss_gamma_2 < loss_gamma_0
        assert loss_gamma_5 < loss_gamma_2

    def test_alpha_weighting(self, sample_inputs, sample_targets):
        """Test alpha parameter for class weighting."""
        loss_no_alpha = FocalLoss(alpha=None, gamma=2.0)(sample_inputs, sample_targets)
        loss_low_alpha = FocalLoss(alpha=0.25, gamma=2.0)(sample_inputs, sample_targets)
        loss_high_alpha = FocalLoss(alpha=0.75, gamma=2.0)(
            sample_inputs, sample_targets
        )

        # Different alpha should produce different losses
        assert not torch.isclose(loss_low_alpha, loss_high_alpha)

    def test_reduction_mean(self, sample_inputs, sample_targets):
        """Test mean reduction (default)."""
        loss_fn = FocalLoss(reduction="mean")
        loss = loss_fn(sample_inputs, sample_targets)

        assert loss.dim() == 0  # scalar

    def test_reduction_sum(self, sample_inputs, sample_targets):
        """Test sum reduction."""
        loss_fn = FocalLoss(reduction="sum")
        loss = loss_fn(sample_inputs, sample_targets)

        assert loss.dim() == 0  # scalar

        # Sum should be larger than mean (for n > 1)
        loss_mean = FocalLoss(reduction="mean")(sample_inputs, sample_targets)
        assert loss > loss_mean

    def test_reduction_none(self, sample_inputs, sample_targets):
        """Test no reduction (per-sample loss)."""
        loss_fn = FocalLoss(reduction="none")
        loss = loss_fn(sample_inputs, sample_targets)

        assert loss.shape == sample_inputs.shape
        assert (loss >= 0).all()

    def test_perfect_prediction(self):
        """Test loss with perfect predictions (very confident correct)."""
        # Very high logits for positive class
        inputs = torch.tensor([10.0, 10.0, 10.0])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss_fn = FocalLoss()
        loss = loss_fn(inputs, targets)

        # Easy examples (high p_t) should have very low focal loss
        assert loss.item() < 0.001

    def test_worst_prediction(self):
        """Test loss with worst predictions (confident but wrong)."""
        # Very negative logits for positive class (confident wrong prediction)
        inputs = torch.tensor([-10.0, -10.0, -10.0])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss_fn = FocalLoss(alpha=None, gamma=0.0)  # Use gamma=0 for comparison
        loss = loss_fn(inputs, targets)

        assert loss.item() > 5  # Should be large

    def test_hard_example_focus(self):
        """Test that focal loss focuses on hard examples."""
        # Mix of easy and hard examples
        # Easy: high confidence correct prediction
        # Hard: low confidence prediction
        inputs = torch.tensor([5.0, 0.1])  # First is easy, second is hard
        targets = torch.tensor([1.0, 1.0])

        loss_none = FocalLoss(alpha=None, gamma=2.0, reduction="none")(inputs, targets)

        # Hard example should have higher loss than easy example
        assert loss_none[1] > loss_none[0]

    def test_2d_input(self):
        """Test with 2D input (N, 1) shape."""
        inputs = torch.tensor([[0.5], [-0.3], [1.2]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])

        loss_fn = FocalLoss()
        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_gradient_flow(self, sample_inputs, sample_targets):
        """Test that gradients can flow through the loss."""
        inputs = sample_inputs.clone().requires_grad_(True)

        loss_fn = FocalLoss()
        loss = loss_fn(inputs, sample_targets)
        loss.backward()

        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large positive logits
        inputs_large = torch.tensor([100.0, 100.0])
        targets = torch.tensor([1.0, 0.0])

        loss_fn = FocalLoss()
        loss = loss_fn(inputs_large, targets)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Very large negative logits
        inputs_small = torch.tensor([-100.0, -100.0])
        loss_small = loss_fn(inputs_small, targets)

        assert not torch.isnan(loss_small)
        assert not torch.isinf(loss_small)

    def test_device_compatibility(self, sample_inputs, sample_targets):
        """Test on CPU device."""
        loss_fn = FocalLoss()
        loss = loss_fn(sample_inputs, sample_targets)

        assert loss.device.type == "cpu"

    def test_different_alpha_values(self, sample_inputs, sample_targets):
        """Test various alpha values."""
        alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]

        losses = []
        for alpha in alpha_values:
            loss = FocalLoss(alpha=alpha)(sample_inputs, sample_targets)
            losses.append(loss.item())
            assert loss.item() > 0

        # Check that different alpha produces different losses
        assert len(set(losses)) == len(losses)  # All unique

    def test_different_gamma_values(self, sample_inputs, sample_targets):
        """Test various gamma values."""
        gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]

        losses = []
        for gamma in gamma_values:
            loss = FocalLoss(alpha=None, gamma=gamma)(sample_inputs, sample_targets)
            losses.append(loss.item())
            assert loss.item() > 0

        # Higher gamma should generally reduce loss (down-weights easy examples)
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1] - 0.01  # Allow small tolerance

    def test_batch_consistency(self):
        """Test that batching doesn't affect mean loss."""
        inputs = torch.tensor([0.5, -0.3, 1.2, -0.8])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Full batch
        loss_full = FocalLoss(reduction="mean")(inputs, targets)

        # Compute mean of individual losses
        loss_individual = FocalLoss(reduction="none")(inputs, targets)
        loss_manual_mean = loss_individual.mean()

        assert torch.isclose(loss_full, loss_manual_mean, atol=1e-6)

    def test_imbalanced_dataset_simulation(self):
        """Test focal loss behavior on imbalanced data."""
        # Simulate imbalanced: 90% negative, 10% positive
        torch.manual_seed(42)
        n_samples = 100
        n_positive = 10

        inputs = torch.randn(n_samples)
        targets = torch.zeros(n_samples)
        targets[:n_positive] = 1.0

        # Focal loss should handle imbalance better than BCE
        focal_loss = FocalLoss(alpha=0.75, gamma=2.0)(inputs, targets)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # Both should produce valid losses
        assert not torch.isnan(focal_loss)
        assert not torch.isnan(bce_loss)
        assert focal_loss.item() > 0
        assert bce_loss.item() > 0

    def test_symmetric_predictions(self):
        """Test with symmetric predictions around 0.5."""
        inputs = torch.tensor([0.0, 0.0, 0.0, 0.0])  # p = 0.5
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)  # Use balanced alpha
        loss_none = loss_fn.__class__(alpha=0.5, gamma=2.0, reduction="none")
        losses = loss_none(inputs, targets)

        # All predictions are equally uncertain, so focal weights should be equal
        # But alpha affects positive vs negative differently
        assert losses.shape == inputs.shape
