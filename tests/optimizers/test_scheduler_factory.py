"""Unit tests for scheduler factory."""

import pytest
import torch
import torch.nn as nn

from nkululeko.optimizers.scheduler_factory import (
    get_scheduler,
    initialize_cosine_scheduler,
    step_scheduler,
)


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
def simple_optimizer():
    """Create a simple optimizer for testing."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return optimizer


class TestGetScheduler:
    """Tests for get_scheduler function."""

    def test_cosine_scheduler_lazy_init(self, simple_optimizer):
        """Test that cosine scheduler returns None and needs lazy init."""
        util = MockUtil({})
        scheduler, sched_type, needs_init = get_scheduler(
            simple_optimizer, util, default_scheduler="cosine"
        )

        assert scheduler is None
        assert sched_type == "cosine"
        assert needs_init is True
        assert any("cosine" in msg for msg in util.debug_messages)

    def test_step_scheduler(self, simple_optimizer):
        """Test creating step scheduler."""
        util = MockUtil(
            {
                "MODEL.scheduler": "step",
                "MODEL.scheduler.step_size": "10",
                "MODEL.scheduler.gamma": "0.5",
            }
        )
        scheduler, sched_type, needs_init = get_scheduler(simple_optimizer, util)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert sched_type == "step"
        assert needs_init is False

    def test_exponential_scheduler(self, simple_optimizer):
        """Test creating exponential scheduler."""
        util = MockUtil(
            {"MODEL.scheduler": "exponential", "MODEL.scheduler.gamma": "0.95"}
        )
        scheduler, sched_type, needs_init = get_scheduler(simple_optimizer, util)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert sched_type == "exponential"
        assert needs_init is False

    def test_none_scheduler(self, simple_optimizer):
        """Test no scheduler."""
        util = MockUtil({"MODEL.scheduler": "none"})
        scheduler, sched_type, needs_init = get_scheduler(simple_optimizer, util)

        assert scheduler is None
        assert sched_type == "none"
        assert needs_init is False

    def test_false_scheduler(self, simple_optimizer):
        """Test scheduler=false (alternative to none)."""
        util = MockUtil({"MODEL.scheduler": "false"})
        scheduler, sched_type, needs_init = get_scheduler(simple_optimizer, util)

        assert scheduler is None
        assert sched_type == "none"
        assert needs_init is False

    def test_scheduler_case_insensitive(self, simple_optimizer):
        """Test that scheduler type is case insensitive."""
        util_upper = MockUtil({"MODEL.scheduler": "STEP"})
        scheduler_upper, _, _ = get_scheduler(simple_optimizer, util_upper)

        util_mixed = MockUtil({"MODEL.scheduler": "StEp"})
        scheduler_mixed, _, _ = get_scheduler(simple_optimizer, util_mixed)

        assert isinstance(scheduler_upper, torch.optim.lr_scheduler.StepLR)
        assert isinstance(scheduler_mixed, torch.optim.lr_scheduler.StepLR)

    def test_unknown_scheduler_raises_error(self, simple_optimizer):
        """Test that unknown scheduler type raises error."""
        util = MockUtil({"MODEL.scheduler": "unknown_scheduler"})

        with pytest.raises(ValueError, match="unknown scheduler type"):
            get_scheduler(simple_optimizer, util)


class TestInitializeCosineScheduler:
    """Tests for initialize_cosine_scheduler function."""

    def test_cosine_scheduler_initialization(self, simple_optimizer):
        """Test basic cosine scheduler initialization."""
        util = MockUtil({})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=100, total_epochs=50
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_warmup_phase(self, simple_optimizer):
        """Test that warmup phase increases learning rate linearly."""
        util = MockUtil({"MODEL.warmup_epochs": "5"})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=100, total_epochs=50
        )

        # At step 0, LR should be 0 (or very small)
        lr_multipliers = []
        for step in [0, 100, 250, 500]:  # 0, 1, 2.5, 5 epochs
            # Get LR multiplier at this step
            lr_mult = scheduler.lr_lambdas[0](step)
            lr_multipliers.append(lr_mult)

        # Should increase during warmup
        assert lr_multipliers[0] < lr_multipliers[1]
        assert lr_multipliers[1] < lr_multipliers[2]
        # Should reach ~1.0 at end of warmup
        assert abs(lr_multipliers[3] - 1.0) < 0.01

    def test_cosine_annealing_phase(self, simple_optimizer):
        """Test that cosine annealing decreases learning rate."""
        util = MockUtil({"MODEL.warmup_epochs": "2"})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=100, total_epochs=20
        )

        # After warmup, LR should decrease following cosine curve
        warmup_end = 200  # 2 epochs * 100 steps
        mid_training = 1000  # 10 epochs
        end_training = 2000  # 20 epochs

        lr_warmup = scheduler.lr_lambdas[0](warmup_end)
        lr_mid = scheduler.lr_lambdas[0](mid_training)
        lr_end = scheduler.lr_lambdas[0](end_training)

        # LR should decrease after warmup
        assert lr_warmup > lr_mid > lr_end
        # At end, should be close to 0
        assert lr_end < 0.1

    def test_reads_from_config(self, simple_optimizer):
        """Test that it reads epochs from config if not provided."""
        util = MockUtil({"EXP.epochs": "100", "MODEL.warmup_epochs": "10"})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=50
        )

        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        assert any("100" in msg for msg in util.debug_messages)

    def test_custom_warmup_epochs(self, simple_optimizer):
        """Test custom warmup epochs."""
        util = MockUtil({"MODEL.warmup_epochs": "10"})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=100, total_epochs=50
        )

        # At 10 epochs (1000 steps), should reach full LR
        lr_mult = scheduler.lr_lambdas[0](1000)
        assert abs(lr_mult - 1.0) < 0.01


class TestStepScheduler:
    """Tests for step_scheduler function."""

    def test_step_cosine_per_batch(self, simple_optimizer):
        """Test stepping cosine scheduler per batch."""
        util = MockUtil({})
        scheduler = initialize_cosine_scheduler(
            simple_optimizer, util, steps_per_epoch=100, total_epochs=10
        )

        # Step scheduler twice
        step_scheduler(scheduler, "cosine", step_per_batch=True)
        step_scheduler(scheduler, "cosine", step_per_batch=True)

        # Verify stepping occurred (internal state changed)
        # PyTorch schedulers start at _step_count=1 after init, so after 2 steps it's 3
        assert scheduler._step_count == 3  # noqa: SLF001

    def test_step_step_scheduler_per_epoch(self, simple_optimizer):
        """Test stepping step scheduler per epoch."""
        # Use step_size=1 so LR changes after just 1 epoch
        util = MockUtil({"MODEL.scheduler": "step", "MODEL.scheduler.step_size": "1"})
        scheduler, sched_type, _ = get_scheduler(simple_optimizer, util)

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # Step per batch should NOT change LR for step scheduler
        for _ in range(100):
            step_scheduler(scheduler, sched_type, step_per_batch=True)
        lr_after_batch_steps = simple_optimizer.param_groups[0]["lr"]
        assert lr_after_batch_steps == initial_lr

        # Step per epoch SHOULD change LR for step scheduler (with step_size=1)
        step_scheduler(scheduler, sched_type, step_per_batch=False)
        lr_after_epoch_step = simple_optimizer.param_groups[0]["lr"]
        assert lr_after_epoch_step != initial_lr

    def test_step_none_scheduler(self, simple_optimizer):
        """Test stepping with no scheduler (should be no-op)."""
        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # Should not raise error
        step_scheduler(None, "none", step_per_batch=True)
        step_scheduler(None, "none", step_per_batch=False)

        # LR should not change
        assert simple_optimizer.param_groups[0]["lr"] == initial_lr

    def test_step_exponential_per_epoch(self, simple_optimizer):
        """Test stepping exponential scheduler per epoch."""
        util = MockUtil({"MODEL.scheduler": "exponential"})
        scheduler, sched_type, _ = get_scheduler(simple_optimizer, util)

        initial_lr = simple_optimizer.param_groups[0]["lr"]

        # Step per epoch should decrease LR
        step_scheduler(scheduler, sched_type, step_per_batch=False)
        lr_after = simple_optimizer.param_groups[0]["lr"]

        assert lr_after < initial_lr
