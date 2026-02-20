# scheduler_factory.py
"""Factory for creating learning rate schedulers from config."""

import math

import torch


def get_scheduler(optimizer, util, default_scheduler="cosine"):
    """Create a learning rate scheduler from configuration.

    Reads scheduler type and hyperparameters from the MODEL section of config.

    Supported schedulers:
    - cosine: Cosine annealing with warmup (default)
    - step: Step decay scheduler
    - exponential: Exponential decay scheduler
    - none: No scheduler

    Args:
        optimizer: PyTorch optimizer to attach scheduler to
        util: Utility object for accessing config
        default_scheduler: Default scheduler type if not specified in config

    Returns:
        tuple: (scheduler, scheduler_type, needs_lazy_init)
            - scheduler: torch.optim.lr_scheduler or None
            - scheduler_type: str indicating scheduler type
            - needs_lazy_init: bool, True if scheduler needs initialization in train()

    Config parameters:
        MODEL.scheduler: Scheduler type (default: cosine)
        MODEL.warmup_epochs: Warmup epochs for cosine scheduler (default: 5)
        MODEL.scheduler.step_size: Step size for step scheduler (default: 10)
        MODEL.scheduler.gamma: Gamma for step/exponential scheduler (default: 0.5)
    """
    scheduler_type = util.config_val("MODEL", "scheduler", default_scheduler).lower()

    if scheduler_type == "cosine":
        # Cosine scheduler with warmup needs total steps, so defer initialization
        util.debug("using cosine scheduler with warmup (initialized on first train)")
        return None, "cosine", True

    elif scheduler_type == "step":
        step_size = int(util.config_val("MODEL", "scheduler.step_size", "10"))
        gamma = float(util.config_val("MODEL", "scheduler.gamma", "0.5"))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        util.debug(f"using step scheduler: step_size={step_size}, gamma={gamma}")
        return scheduler, "step", False

    elif scheduler_type == "exponential":
        gamma = float(util.config_val("MODEL", "scheduler.gamma", "0.95"))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        util.debug(f"using exponential scheduler: gamma={gamma}")
        return scheduler, "exponential", False

    elif scheduler_type == "none" or scheduler_type == "false":
        util.debug("no learning rate scheduler")
        return None, "none", False

    else:
        util.error(f"unknown scheduler type: {scheduler_type}")
        # Fallback return in case error doesn't raise exception
        return None, "none", False


def initialize_cosine_scheduler(optimizer, util, steps_per_epoch, total_epochs=None):
    """Initialize cosine scheduler with warmup.

    This must be called during first training epoch when total steps are known.

    Args:
        optimizer: PyTorch optimizer
        util: Utility object for accessing config
        steps_per_epoch: Number of batches per epoch
        total_epochs: Total number of epochs (reads from config if None)

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured scheduler
    """
    if total_epochs is None:
        total_epochs = int(util.config_val("EXP", "epochs", "50"))

    warmup_epochs = int(util.config_val("MODEL", "warmup_epochs", "5"))
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        """Learning rate multiplier as a function of training step."""
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        # Cosine annealing
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    util.debug(
        f"initialized cosine scheduler: total_epochs={total_epochs}, "
        f"warmup_epochs={warmup_epochs}, steps_per_epoch={steps_per_epoch}"
    )
    return scheduler


def step_scheduler(scheduler, scheduler_type, step_per_batch=True):
    """Step the scheduler at appropriate time.

    Args:
        scheduler: PyTorch scheduler or None
        scheduler_type: Type of scheduler ('cosine', 'step', 'exponential', 'none')
        step_per_batch: If True, step after each batch; if False, step after epoch

    Returns:
        None
    """
    if scheduler is None:
        return

    # Cosine scheduler steps per batch
    if scheduler_type == "cosine" and step_per_batch:
        scheduler.step()
    # Step/exponential schedulers step per epoch
    elif scheduler_type in ["step", "exponential"] and not step_per_batch:
        scheduler.step()
