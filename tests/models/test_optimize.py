"""Model._optimize(): the one optimization step shared by the neural models,
with a plain optimizer or SAM (MODEL.sam)."""

import pytest
import torch
import torch.nn as nn

from nkululeko.models.model import Model
from nkululeko.optimizers.sam import SAM


def _setup(seed=0):
    torch.manual_seed(seed)
    net = nn.Linear(3, 1)
    x = torch.randn(8, 3)
    y = x.sum(dim=1, keepdim=True)
    return net, x, y


def _loss_fn(net, x, y, calls=None):
    def compute_loss():
        if calls is not None:
            calls.append(1)
        return nn.functional.mse_loss(net(x), y)

    return compute_loss


@pytest.fixture
def model():
    return Model.__new__(Model)


class TestPlainOptimizer:
    def test_updates_weights_and_returns_the_loss(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        expected = nn.functional.mse_loss(net(x), y).item()
        before = net.weight.detach().clone()
        loss = model._optimize(opt, _loss_fn(net, x, y))
        assert loss.item() == pytest.approx(expected)
        assert not torch.equal(net.weight, before)

    def test_computes_the_loss_once(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        calls = []
        model._optimize(opt, _loss_fn(net, x, y, calls))
        assert len(calls) == 1

    def test_clips_the_gradient(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        before = [p.detach().clone() for p in net.parameters()]
        model._optimize(
            opt,
            _loss_fn(net, x, y * 100),
            clip_params=list(net.parameters()),
            max_grad_norm=0.5,
        )
        moved = torch.sqrt(
            sum(((p - b) ** 2).sum() for p, b in zip(net.parameters(), before))
        )
        assert moved.item() == pytest.approx(0.5, rel=1e-4)  # lr 1 * clipped norm

    def test_no_clipping_by_default(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        before = [p.detach().clone() for p in net.parameters()]
        model._optimize(opt, _loss_fn(net, x, y * 100))
        moved = torch.sqrt(
            sum(((p - b) ** 2).sum() for p, b in zip(net.parameters(), before))
        )
        assert moved.item() > 5

    def test_skip_nan_returns_none_and_leaves_weights(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        before = net.weight.detach().clone()
        loss = model._optimize(
            opt, lambda: net(x).sum() * float("nan"), skip_nan=True
        )
        assert loss is None
        assert torch.equal(net.weight, before)

    def test_nan_is_not_skipped_unless_asked(self, model):
        net, x, y = _setup()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        loss = model._optimize(opt, lambda: net(x).sum() * float("nan"))
        assert torch.isnan(loss)


class TestSam:
    def _sam(self, net, **kw):
        return SAM(net.parameters(), torch.optim.SGD, rho=0.1, lr=0.1, **kw)

    def test_computes_the_loss_twice(self, model):
        net, x, y = _setup()
        calls = []
        model._optimize(self._sam(net), _loss_fn(net, x, y, calls))
        assert len(calls) == 2

    def test_returns_the_loss_at_the_current_weights(self, model):
        net, x, y = _setup()
        expected = nn.functional.mse_loss(net(x), y).item()
        loss = model._optimize(self._sam(net), _loss_fn(net, x, y))
        assert loss.item() == pytest.approx(expected)

    def test_differs_from_a_plain_step(self, model):
        net_a, x, y = _setup()
        net_b, _, _ = _setup()
        model._optimize(self._sam(net_a), _loss_fn(net_a, x, y))
        model._optimize(
            torch.optim.SGD(net_b.parameters(), lr=0.1), _loss_fn(net_b, x, y)
        )
        assert not torch.allclose(net_a.weight, net_b.weight)

    def test_clips_both_passes(self, model):
        net, x, y = _setup()
        opt = SAM(net.parameters(), torch.optim.SGD, rho=0.0, lr=1.0)
        before = [p.detach().clone() for p in net.parameters()]
        model._optimize(
            opt,
            _loss_fn(net, x, y * 100),
            clip_params=list(net.parameters()),
            max_grad_norm=0.5,
        )
        moved = torch.sqrt(
            sum(((p - b) ** 2).sum() for p, b in zip(net.parameters(), before))
        )
        assert moved.item() == pytest.approx(0.5, rel=1e-4)

    def test_skip_nan_returns_none_and_leaves_weights(self, model):
        net, x, y = _setup()
        before = net.weight.detach().clone()
        loss = model._optimize(
            self._sam(net), lambda: net(x).sum() * float("nan"), skip_nan=True
        )
        assert loss is None
        assert torch.equal(net.weight, before)

    def test_nan_never_reaches_the_weights(self, model):
        net, x, y = _setup()
        model._optimize(self._sam(net), lambda: net(x).sum() * float("nan"))
        assert torch.isfinite(net.weight).all()

    def test_trains_with_a_scheduler(self, model):
        net, x, y = _setup()
        opt = self._sam(net)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        first = model._optimize(opt, _loss_fn(net, x, y)).item()
        for _ in range(30):
            sched.step()
            last = model._optimize(opt, _loss_fn(net, x, y)).item()
        assert last < first
