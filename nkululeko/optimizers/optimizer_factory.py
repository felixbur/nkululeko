# optimizer_factory.py
"""Factory for creating optimizers from config."""

import torch

from nkululeko.optimizers.sam import SAM


def get_optimizer(model_parameters, util, default_lr=0.0001, default_optimizer="adamw"):
    """Create an optimizer from configuration.

    Reads optimizer type and hyperparameters from the MODEL section of config.

    Supported optimizers:
    - adamw: AdamW optimizer with weight decay (default)
    - adam: Adam optimizer
    - sgd: SGD optimizer with momentum

    Args:
        model_parameters: Model parameters to optimize (e.g., model.parameters())
        util: Utility object for accessing config
        default_lr: Default learning rate if not specified in config
        default_optimizer: Default optimizer type if not specified in config

    Returns:
        torch.optim.Optimizer: Configured optimizer, wrapped in
        nkululeko.optimizers.sam.SAM if MODEL.sam is set

    Config parameters:
        MODEL.learning_rate: Learning rate (default: 0.0001)
        MODEL.optimizer: Optimizer type - adam, adamw, or sgd (default: adamw)
        MODEL.weight_decay: Weight decay for AdamW (default: 0.01)
        MODEL.momentum: Momentum for SGD (default: 0.9)
        MODEL.sam: Wrap the optimizer in Sharpness-Aware Minimization
            (default: False). Needs two forward/backward passes per step, so
            a model's train loop must use Model._optimize().
        MODEL.sam_rho: SAM neighborhood size (default: 0.05)
    """
    learning_rate = float(util.config_val("MODEL", "learning_rate", str(default_lr)))
    optimizer_type = util.config_val("MODEL", "optimizer", default_optimizer).lower()
    use_sam = util.config_val_bool("MODEL", "sam", False)

    if optimizer_type == "adamw":
        weight_decay = float(util.config_val("MODEL", "weight_decay", "0.01"))
        base_optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
        log_extra = f", weight_decay={weight_decay}"

    elif optimizer_type == "adam":
        base_optimizer_cls = torch.optim.Adam
        optimizer_kwargs = {"lr": learning_rate}
        log_extra = ""

    elif optimizer_type == "sgd":
        momentum = float(util.config_val("MODEL", "momentum", "0.9"))
        base_optimizer_cls = torch.optim.SGD
        optimizer_kwargs = {"lr": learning_rate, "momentum": momentum}
        log_extra = f", momentum={momentum}"

    else:
        util.error(f"unknown optimizer: {optimizer_type}")
        # Fallback return in case error doesn't raise exception
        raise ValueError(f"unknown optimizer: {optimizer_type}")

    if use_sam:
        sam_rho = util.get_sam_rho()
        optimizer = SAM(
            model_parameters, base_optimizer_cls, rho=sam_rho, **optimizer_kwargs
        )
        util.debug(
            f"using {optimizer_type} optimizer wrapped in SAM (rho={sam_rho}): "
            f"lr={learning_rate}{log_extra}"
        )
    else:
        optimizer = base_optimizer_cls(model_parameters, **optimizer_kwargs)
        util.debug(f"using {optimizer_type} optimizer: lr={learning_rate}{log_extra}")

    return optimizer, learning_rate
