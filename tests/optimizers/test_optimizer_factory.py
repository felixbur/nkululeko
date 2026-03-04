"""Unit tests for optimizer factory."""

import pytest
import torch
import torch.nn as nn

from nkululeko.optimizers.optimizer_factory import get_optimizer


class MockUtil:
    """Mock utility class for testing."""

    def __init__(self, config_dict):
        self.config = config_dict
        self.debug_messages = []

    def config_val(self, section, key, default):
        """Mock config_val method."""
        return self.config.get(f"{section}.{key}", default)

    def debug(self, message):
        """Mock debug method."""
        self.debug_messages.append(message)

    def error(self, message):
        """Mock error method."""
        raise ValueError(message)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Linear(10, 5)
    return model


class TestGetOptimizer:
    """Tests for get_optimizer function."""

    def test_adam_optimizer_default(self, simple_model):
        """Test creating Adam optimizer with default settings."""
        util = MockUtil({})
        optimizer, lr = get_optimizer(
            simple_model.parameters(), util, default_lr=0.001, default_optimizer="adam"
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert lr == 0.001
        assert len(util.debug_messages) > 0
        assert "adam" in util.debug_messages[0].lower()

    def test_adam_optimizer_custom_lr(self, simple_model):
        """Test Adam optimizer with custom learning rate."""
        util = MockUtil({"MODEL.learning_rate": "0.0005"})
        optimizer, lr = get_optimizer(
            simple_model.parameters(), util, default_lr=0.001, default_optimizer="adam"
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert lr == 0.0005

    def test_adamw_optimizer(self, simple_model):
        """Test creating AdamW optimizer."""
        util = MockUtil({"MODEL.optimizer": "adamw", "MODEL.weight_decay": "0.01"})
        optimizer, lr = get_optimizer(
            simple_model.parameters(), util, default_lr=0.0001
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert any("weight_decay" in msg for msg in util.debug_messages)

    def test_adamw_custom_weight_decay(self, simple_model):
        """Test AdamW with custom weight decay."""
        util = MockUtil(
            {
                "MODEL.optimizer": "adamw",
                "MODEL.learning_rate": "0.0001",
                "MODEL.weight_decay": "0.05",
            }
        )
        optimizer, lr = get_optimizer(simple_model.parameters(), util)

        assert isinstance(optimizer, torch.optim.AdamW)
        # Check weight_decay is set (access from param_groups)
        assert optimizer.param_groups[0]["weight_decay"] == 0.05

    def test_sgd_optimizer(self, simple_model):
        """Test creating SGD optimizer."""
        util = MockUtil({"MODEL.optimizer": "sgd", "MODEL.momentum": "0.9"})
        optimizer, lr = get_optimizer(simple_model.parameters(), util, default_lr=0.01)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_sgd_custom_momentum(self, simple_model):
        """Test SGD with custom momentum."""
        util = MockUtil({"MODEL.optimizer": "sgd", "MODEL.momentum": "0.95"})
        optimizer, lr = get_optimizer(simple_model.parameters(), util)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.95

    def test_optimizer_case_insensitive(self, simple_model):
        """Test that optimizer type is case insensitive."""
        util_upper = MockUtil({"MODEL.optimizer": "ADAM"})
        optimizer_upper, _ = get_optimizer(simple_model.parameters(), util_upper)

        util_mixed = MockUtil({"MODEL.optimizer": "AdAm"})
        optimizer_mixed, _ = get_optimizer(simple_model.parameters(), util_mixed)

        assert isinstance(optimizer_upper, torch.optim.Adam)
        assert isinstance(optimizer_mixed, torch.optim.Adam)

    def test_unknown_optimizer_raises_error(self, simple_model):
        """Test that unknown optimizer type raises error."""
        util = MockUtil({"MODEL.optimizer": "unknown_optimizer"})

        with pytest.raises(ValueError, match="unknown optimizer"):
            get_optimizer(simple_model.parameters(), util)

    def test_learning_rate_returned(self, simple_model):
        """Test that learning rate is correctly returned."""
        util = MockUtil({"MODEL.learning_rate": "0.005"})
        _, lr = get_optimizer(simple_model.parameters(), util)

        assert lr == 0.005

    def test_optimizer_parameters_linked(self, simple_model):
        """Test that optimizer is linked to model parameters."""
        util = MockUtil({})
        optimizer, _ = get_optimizer(
            simple_model.parameters(), util, default_optimizer="adam"
        )

        # Check that optimizer has the same number of parameter groups
        assert len(optimizer.param_groups) > 0
        # Perform a dummy optimization step to verify linkage
        optimizer.zero_grad()
        loss = simple_model(torch.randn(2, 10)).sum()
        loss.backward()
        optimizer.step()
        # If we get here without errors, parameters are correctly linked

    def test_debug_messages_logged(self, simple_model):
        """Test that debug messages are logged for each optimizer."""
        util_adam = MockUtil({"MODEL.optimizer": "adam"})
        get_optimizer(simple_model.parameters(), util_adam)
        assert len(util_adam.debug_messages) > 0

        util_adamw = MockUtil({"MODEL.optimizer": "adamw"})
        get_optimizer(simple_model.parameters(), util_adamw)
        assert len(util_adamw.debug_messages) > 0

        util_sgd = MockUtil({"MODEL.optimizer": "sgd"})
        get_optimizer(simple_model.parameters(), util_sgd)
        assert len(util_sgd.debug_messages) > 0
