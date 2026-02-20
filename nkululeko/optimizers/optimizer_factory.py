# optimizer_factory.py
"""Factory for creating optimizers from config."""

import torch


def get_optimizer(model_parameters, util, default_lr=0.0001, default_optimizer="adam"):
    """Create an optimizer from configuration.

    Reads optimizer type and hyperparameters from the MODEL section of config.

    Supported optimizers:
    - adam: Adam optimizer (default)
    - adamw: AdamW optimizer with weight decay
    - sgd: SGD optimizer with momentum

    Args:
        model_parameters: Model parameters to optimize (e.g., model.parameters())
        util: Utility object for accessing config
        default_lr: Default learning rate if not specified in config
        default_optimizer: Default optimizer type if not specified in config

    Returns:
        torch.optim.Optimizer: Configured optimizer

    Config parameters:
        MODEL.learning_rate: Learning rate (default: 0.0001)
        MODEL.optimizer: Optimizer type - adam, adamw, or sgd (default: adam)
        MODEL.weight_decay: Weight decay for AdamW (default: 0.01)
        MODEL.momentum: Momentum for SGD (default: 0.9)
    """
    learning_rate = float(util.config_val("MODEL", "learning_rate", str(default_lr)))
    optimizer_type = util.config_val("MODEL", "optimizer", default_optimizer).lower()

    if optimizer_type == "adamw":
        weight_decay = float(util.config_val("MODEL", "weight_decay", "0.01"))
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        util.debug(
            f"using {optimizer_type} optimizer: lr={learning_rate}, weight_decay={weight_decay}"
        )

    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
        util.debug(f"using {optimizer_type} optimizer: lr={learning_rate}")

    elif optimizer_type == "sgd":
        momentum = float(util.config_val("MODEL", "momentum", "0.9"))
        optimizer = torch.optim.SGD(
            model_parameters, lr=learning_rate, momentum=momentum
        )
        util.debug(
            f"using {optimizer_type} optimizer: lr={learning_rate}, momentum={momentum}"
        )

    else:
        util.error(f"unknown optimizer: {optimizer_type}")
        # Fallback return in case error doesn't raise exception
        raise ValueError(f"unknown optimizer: {optimizer_type}")

    return optimizer, learning_rate
