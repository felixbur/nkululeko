"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
Generalization" (ICLR 2021, arXiv:2010.01412). Instead of descending from
the current weights, SAM first moves to the worst-case point in a ball of
radius rho around them, computes the gradient there, and then descends
from the *original* weights with that gradient. This biases training
toward flat regions of the loss landscape, which are reported to
generalize better to unseen domains.

SAM wraps a base optimizer (Adam, AdamW, SGD, ...) rather than replacing
it: the ascent step does not depend on the base optimizer, only the final
descent step uses the base optimizer's own update rule.
"""

import contextlib

import torch


class SAM(torch.optim.Optimizer):
    """Wraps `base_optimizer_cls` with SAM's two-step ascent/descent rule.

    Unlike a plain optimizer, `step()` needs a *closure* that zeroes the
    gradients, recomputes the loss, calls `loss.backward()` and returns the
    loss, because the loss and gradient are computed twice per training
    step: once at the current weights (to find the ascent direction) and
    once at the perturbed weights (the gradient that is actually applied).
    `Model._optimize()` builds that closure for the models.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls,
        rho: float = 0.05,
        adaptive: bool = False,
        **base_optimizer_kwargs,
    ):
        if rho < 0:
            raise ValueError(f"SAM rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(
            self.param_groups, **base_optimizer_kwargs
        )
        # share the groups so that learning-rate schedulers changing
        # `optimizer.param_groups` also change the base optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Move to the worst-case point in the rho-ball (changes the weights
        in place; the weights before the move are kept in
        `self.state[p]["old_p"]` for `second_step()`)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore the weights from before `first_step()`, then let the base
        optimizer update them with the gradient computed at the perturbed
        point (the backward pass must run between the two steps)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure):
        """One SAM step. Returns the loss at the current weights (the first
        closure call). If that loss or its gradient is not finite the step is
        skipped: a NaN/inf gradient norm would otherwise turn every weight
        into NaN."""
        if closure is None:
            raise ValueError(
                "SAM.step() requires a closure (zero_grad, forward, "
                "loss.backward(), return loss): it runs twice per step"
            )
        closure = torch.enable_grad()(closure)
        loss = closure()
        if not (torch.isfinite(loss) and torch.isfinite(self._grad_norm())):
            self.zero_grad()
            return loss
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@contextlib.contextmanager
def frozen_batchnorm_stats(modules):
    """Stop the BatchNorm layers inside `modules` from updating their running
    statistics (momentum 0) while the block runs. SAM computes the loss twice
    per step; only the first pass, at the real weights, should feed the
    statistics."""
    norms = [
        layer
        for module in modules
        for layer in module.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
    ]
    momenta = [layer.momentum for layer in norms]
    for layer in norms:
        layer.momentum = 0.0
    try:
        yield
    finally:
        for layer, momentum in zip(norms, momenta):
            layer.momentum = momentum


def is_sam_optimizer(optimizer) -> bool:
    """True if `get_optimizer()` wrapped the base optimizer in SAM
    (`MODEL.sam = True`)."""
    return isinstance(optimizer, SAM)
