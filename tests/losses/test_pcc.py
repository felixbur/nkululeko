"""Unit tests for Pearson Correlation Coefficient (PCC) loss."""

import pytest
import torch

from nkululeko.losses.loss_pcc import PearsonCorCoeff


@pytest.fixture
def pcc_loss():
    """Create PCC loss instance."""
    return PearsonCorCoeff()


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    return torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])


class TestPearsonCorCoeff:
    """Tests for PCC loss function."""

    def test_basic_functionality(
        self, pcc_loss, sample_predictions, sample_ground_truth
    ):
        """Test basic PCC loss computation."""
        loss = pcc_loss(sample_predictions, sample_ground_truth)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar output

    def test_perfect_correlation(self, pcc_loss):
        """Test loss with perfect correlation (identical predictions)."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        loss = pcc_loss(predictions, ground_truth)

        # Perfect correlation -> PCC = 1.0 -> loss = 1 - 1 = 0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_high_correlation(self, pcc_loss, sample_predictions, sample_ground_truth):
        """Test loss with high but not perfect correlation."""
        loss = pcc_loss(sample_predictions, sample_ground_truth)

        # High correlation should give low loss (close to 0)
        assert loss.item() < 0.1
        assert loss.item() >= 0

    def test_no_correlation(self, pcc_loss):
        """Test loss with uncorrelated data."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])

        loss = pcc_loss(predictions, ground_truth)

        # Low/no correlation should give higher loss
        assert loss.item() > 0.5

    def test_negative_correlation(self, pcc_loss):
        """Test loss with negative correlation."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        loss = pcc_loss(predictions, ground_truth)

        # Negative correlation -> PCC ~ -1 -> loss = 1 - (-1) = 2
        assert loss.item() > 1.0

    def test_scaled_predictions_unaffected(self, pcc_loss):
        """Pearson correlation is scale-invariant, unlike CCC."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled_predictions = predictions * 2  # Scale by 2

        loss_original = pcc_loss(predictions, ground_truth)
        loss_scaled = pcc_loss(scaled_predictions, ground_truth)

        assert torch.isclose(loss_original, loss_scaled, atol=1e-6)

    def test_shifted_predictions_unaffected(self, pcc_loss):
        """Pearson correlation is shift-invariant, unlike CCC."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        shifted_predictions = predictions + 10  # Shift mean

        loss_original = pcc_loss(predictions, ground_truth)
        loss_shifted = pcc_loss(shifted_predictions, ground_truth)

        assert torch.isclose(loss_original, loss_shifted, atol=1e-6)

    def test_2d_input(self, pcc_loss):
        """Test with 2D input tensors."""
        predictions = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        ground_truth = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        loss = pcc_loss(predictions.squeeze(), ground_truth.squeeze())

        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_batch_computation(self, pcc_loss):
        """Test PCC computation across batch dimension."""
        predictions = torch.tensor([1.5, 2.3, 3.1, 4.2, 5.0, 6.1])
        ground_truth = torch.tensor([1.4, 2.5, 3.0, 4.0, 5.2, 6.0])

        loss = pcc_loss(predictions, ground_truth)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_gradient_flow(self, pcc_loss):
        """Test that gradients can flow through the loss."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        ground_truth = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])

        loss = pcc_loss(predictions, ground_truth)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_loss_range(self, pcc_loss):
        """Test that loss values are in expected range [0, 2]."""
        test_cases = [
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),  # Perfect
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])),  # Negative
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 1.0, 3.0])),  # Mixed
        ]

        for predictions, ground_truth in test_cases:
            loss = pcc_loss(predictions, ground_truth)
            # PCC ranges from -1 to 1, so loss = 1 - PCC ranges from 0 to 2
            assert loss.item() >= -0.1  # Allow small numerical errors
            assert loss.item() <= 2.1

    def test_constant_predictions(self, pcc_loss):
        """Test with constant predictions (zero variance)."""
        predictions = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # This may produce NaN or inf due to division by zero
        # Just check it doesn't crash
        loss = pcc_loss(predictions, ground_truth)
        assert isinstance(loss, torch.Tensor)

    def test_float_precision(self, pcc_loss):
        """Test with different float precisions."""
        predictions_f32 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        ground_truth_f32 = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8], dtype=torch.float32)

        predictions_f64 = predictions_f32.double()
        ground_truth_f64 = ground_truth_f32.double()

        loss_f32 = pcc_loss(predictions_f32, ground_truth_f32)
        loss_f64 = pcc_loss(predictions_f64, ground_truth_f64)

        # Results should be similar regardless of precision
        assert torch.isclose(loss_f32.double(), loss_f64, atol=1e-5)

    def test_integer_ground_truth(self, pcc_loss):
        """model_tuned.py's HF Trainer path may pass integer-dtype labels;
        torch.mean() errors on integer tensors, so ground_truth must be cast
        to float internally rather than requiring the caller to do it."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)

        loss = pcc_loss(predictions, ground_truth)

        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_device_compatibility(self, pcc_loss):
        """Test on CPU device."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
        ground_truth = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8], device="cpu")

        loss = pcc_loss(predictions, ground_truth)

        assert loss.device.type == "cpu"
