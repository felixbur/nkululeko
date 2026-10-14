"""Unit tests for the SAM optimizer wrapper."""

import pytest
import torch
import torch.nn as nn

from nkululeko.optimizers.sam import SAM, is_sam_optimizer


def _problem(seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(4, 1)
    x = torch.randn(16, 4)
    y = x.sum(dim=1, keepdim=True)
    return model, x, y


def _closure(model, optimizer, x, y, calls=None):
    def closure():
        if calls is not None:
            calls.append(1)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    return closure


class TestConstruction:
    def test_wraps_base_optimizer(self):
        model, _, _ = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.1, lr=0.2, momentum=0.9)
        assert isinstance(opt.base_optimizer, torch.optim.SGD)
        assert opt.param_groups is opt.base_optimizer.param_groups
        assert opt.param_groups[0]["lr"] == 0.2
        assert opt.param_groups[0]["rho"] == 0.1
        assert is_sam_optimizer(opt)

    def test_plain_optimizer_is_not_sam(self):
        model, _, _ = _problem()
        assert not is_sam_optimizer(torch.optim.SGD(model.parameters(), lr=0.1))

    def test_negative_rho_rejected(self):
        model, _, _ = _problem()
        with pytest.raises(ValueError, match="rho"):
            SAM(model.parameters(), torch.optim.SGD, rho=-0.1, lr=0.1)

    def test_step_without_closure_rejected(self):
        model, _, _ = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, lr=0.1)
        with pytest.raises(ValueError, match="closure"):
            opt.step(None)


class TestStep:
    def test_closure_runs_twice_per_step(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
        calls = []
        opt.step(_closure(model, opt, x, y, calls))
        assert len(calls) == 2

    def test_returns_loss_at_the_current_weights(self):
        model, x, y = _problem()
        expected = nn.functional.mse_loss(model(x), y).item()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.5, lr=0.1)
        loss = opt.step(_closure(model, opt, x, y))
        assert loss.item() == pytest.approx(expected)

    def test_first_step_perturbs_second_step_restores_and_updates(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.1, lr=0.1)
        start = [p.detach().clone() for p in model.parameters()]

        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        grads = [p.grad.clone() for p in model.parameters()]
        opt.first_step(zero_grad=True)
        moved = [p.detach().clone() for p in model.parameters()]
        # the move has length rho, along the gradient
        norm = torch.sqrt(sum(((m - s) ** 2).sum() for m, s in zip(moved, start)))
        assert norm.item() == pytest.approx(0.1, rel=1e-4)
        g_norm = torch.sqrt(sum((g**2).sum() for g in grads))
        for m, s, g in zip(moved, start, grads):
            assert torch.allclose(m - s, 0.1 * g / g_norm, atol=1e-6)

        nn.functional.mse_loss(model(x), y).backward()
        perturbed_grads = [p.grad.clone() for p in model.parameters()]
        opt.second_step(zero_grad=True)
        # SGD update from the ORIGINAL weights with the perturbed gradient
        for p, s, g in zip(model.parameters(), start, perturbed_grads):
            assert torch.allclose(p, s - 0.1 * g, atol=1e-6)

    def test_rho_zero_is_the_base_optimizer(self):
        model_a, x, y = _problem()
        model_b, _, _ = _problem()
        sam = SAM(model_a.parameters(), torch.optim.SGD, rho=0.0, lr=0.1)
        plain = torch.optim.SGD(model_b.parameters(), lr=0.1)
        for _ in range(3):
            sam.step(_closure(model_a, sam, x, y))
            plain.zero_grad()
            nn.functional.mse_loss(model_b(x), y).backward()
            plain.step()
        for a, b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(a, b, atol=1e-6)

    def test_adaptive_scales_the_move_by_the_weights(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.1, adaptive=True, lr=0.1)
        start = [p.detach().clone() for p in model.parameters()]
        opt.step(_closure(model, opt, x, y))
        assert all(
            torch.isfinite(p).all() and not torch.equal(p, s)
            for p, s in zip(model.parameters(), start)
        )

    def test_trains_a_toy_problem(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=0.05)
        first = opt.step(_closure(model, opt, x, y)).item()
        for _ in range(200):
            last = opt.step(_closure(model, opt, x, y)).item()
        assert last < first * 0.05

    def test_non_finite_loss_skips_the_update(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
        start = [p.detach().clone() for p in model.parameters()]

        def closure():
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y) * float("nan")
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert torch.isnan(loss)
        for p, s in zip(model.parameters(), start):
            assert torch.equal(p, s)
            assert p.grad is None or torch.all(p.grad == 0)

    def test_non_finite_gradient_skips_the_update(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=0.1)
        start = [p.detach().clone() for p in model.parameters()]

        def closure():
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            for p in model.parameters():
                p.grad.fill_(float("inf"))  # finite loss, infinite gradient
            return loss

        opt.step(closure)
        for p, s in zip(model.parameters(), start):
            assert torch.equal(p, s)


class TestWithSchedulers:
    def test_scheduler_changes_the_base_optimizer_lr(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=1.0)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        opt.step(_closure(model, opt, x, y))
        sched.step()
        assert opt.base_optimizer.param_groups[0]["lr"] == pytest.approx(0.5)

    def test_no_scheduler_order_warning(self, recwarn):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, lr=1.0)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        opt.step(_closure(model, opt, x, y))
        sched.step()
        assert not [w for w in recwarn if "lr_scheduler.step()" in str(w.message)]


class TestStateDict:
    def test_load_state_dict_keeps_groups_shared(self):
        model, x, y = _problem()
        opt = SAM(model.parameters(), torch.optim.Adam, rho=0.05, lr=0.01)
        opt.step(_closure(model, opt, x, y))
        state = opt.state_dict()
        other_model, _, _ = _problem(1)
        other = SAM(other_model.parameters(), torch.optim.Adam, rho=0.05, lr=0.5)
        other.load_state_dict(state)
        assert other.base_optimizer.param_groups is other.param_groups
        assert other.param_groups[0]["lr"] == 0.01
