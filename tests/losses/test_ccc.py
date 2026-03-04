"""Unit tests for Concordance Correlation Coefficient (CCC) loss."""

import pytest
import torch

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff


@pytest.fixture
def ccc_loss():
    """Create CCC loss instance."""
    return ConcordanceCorCoeff()


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    return torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])


class TestConcordanceCorCoeff:
    """Tests for CCC loss function."""

    def test_basic_functionality(
        self, ccc_loss, sample_predictions, sample_ground_truth
    ):
        """Test basic CCC loss computation."""
        loss = ccc_loss(sample_predictions, sample_ground_truth)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar output

    def test_perfect_correlation(self, ccc_loss):
        """Test loss with perfect correlation (identical predictions)."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        loss = ccc_loss(predictions, ground_truth)

        # Perfect correlation -> CCC = 1.0 -> loss = 1 - 1 = 0
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_high_correlation(self, ccc_loss, sample_predictions, sample_ground_truth):
        """Test loss with high but not perfect correlation."""
        loss = ccc_loss(sample_predictions, sample_ground_truth)

        # High correlation should give low loss (close to 0)
        assert loss.item() < 0.1
        assert loss.item() >= 0

    def test_no_correlation(self, ccc_loss):
        """Test loss with uncorrelated data."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])

        loss = ccc_loss(predictions, ground_truth)

        # Low/no correlation should give higher loss
        assert loss.item() > 0.5

    def test_negative_correlation(self, ccc_loss):
        """Test loss with negative correlation."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        loss = ccc_loss(predictions, ground_truth)

        # Negative correlation -> CCC ~ -1 -> loss = 1 - (-1) = 2
        assert loss.item() > 1.0

    def test_scaled_predictions(self, ccc_loss):
        """Test that scaling affects CCC (unlike Pearson correlation)."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled_predictions = predictions * 2  # Scale by 2

        loss_original = ccc_loss(predictions, ground_truth)
        loss_scaled = ccc_loss(scaled_predictions, ground_truth)

        # Scaled predictions should have higher loss due to different variance
        assert loss_scaled > loss_original

    def test_shifted_predictions(self, ccc_loss):
        """Test that mean shift affects CCC."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        shifted_predictions = predictions + 10  # Shift mean

        loss_original = ccc_loss(predictions, ground_truth)
        loss_shifted = ccc_loss(shifted_predictions, ground_truth)

        # Shifted predictions should have higher loss due to mean difference
        assert loss_shifted > loss_original

    def test_2d_input(self, ccc_loss):
        """Test with 2D input tensors."""
        predictions = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        ground_truth = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

        loss = ccc_loss(predictions.squeeze(), ground_truth.squeeze())

        assert isinstance(loss, torch.Tensor)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_batch_computation(self, ccc_loss):
        """Test CCC computation across batch dimension."""
        # Simulating batch of samples
        predictions = torch.tensor([1.5, 2.3, 3.1, 4.2, 5.0, 6.1])
        ground_truth = torch.tensor([1.4, 2.5, 3.0, 4.0, 5.2, 6.0])

        loss = ccc_loss(predictions, ground_truth)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_gradient_flow(self, ccc_loss):
        """Test that gradients can flow through the loss."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        ground_truth = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8])

        loss = ccc_loss(predictions, ground_truth)
        loss.backward()

        assert predictions.grad is not None
        assert predictions.grad.shape == predictions.shape

    def test_loss_range(self, ccc_loss):
        """Test that loss values are in expected range [0, 2]."""
        # Test various scenarios
        test_cases = [
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])),  # Perfect
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])),  # Negative
            (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 1.0, 3.0])),  # Mixed
        ]

        for predictions, ground_truth in test_cases:
            loss = ccc_loss(predictions, ground_truth)
            # CCC ranges from -1 to 1, so loss = 1 - CCC ranges from 0 to 2
            assert loss.item() >= -0.1  # Allow small numerical errors
            assert loss.item() <= 2.1

    def test_constant_predictions(self, ccc_loss):
        """Test with constant predictions (zero variance)."""
        predictions = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
        ground_truth = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # This may produce NaN or inf due to division by zero
        # Just check it doesn't crash
        loss = ccc_loss(predictions, ground_truth)
        assert isinstance(loss, torch.Tensor)

    def test_float_precision(self, ccc_loss):
        """Test with different float precisions."""
        predictions_f32 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        ground_truth_f32 = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8], dtype=torch.float32)

        predictions_f64 = predictions_f32.double()
        ground_truth_f64 = ground_truth_f32.double()

        loss_f32 = ccc_loss(predictions_f32, ground_truth_f32)
        loss_f64 = ccc_loss(predictions_f64, ground_truth_f64)

        # Results should be similar regardless of precision
        assert torch.isclose(loss_f32.double(), loss_f64, atol=1e-5)

    def test_device_compatibility(self, ccc_loss):
        """Test on CPU device."""
        predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cpu")
        ground_truth = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.8], device="cpu")

        loss = ccc_loss(predictions, ground_truth)

        assert loss.device.type == "cpu"
