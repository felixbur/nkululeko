"""Unit tests for the model-agnostic DANN components
(nkululeko/models/domain_adversarial.py)."""


import pytest
import torch
import torch.nn as nn

import pandas as pd

from nkululeko.models.domain_adversarial import (
    DannConfig,
    DannHeads,
    DomainAdversarialHead,
    GradientReversalLayer,
)


class TestGradientReversalLayer:
    def test_forward_is_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 8)
        assert torch.equal(grl(x), x)

    def test_backward_negates_and_scales_gradient(self):
        grl = GradientReversalLayer(lambda_=2.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()

        # d(sum(y))/dx would be all-ones without reversal; with
        # lambda_=2.0 it must be all -2.0.
        assert torch.allclose(x.grad, torch.full_like(x, -2.0))

    def test_lambda_is_mutable_between_forward_calls(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x1 = torch.randn(2, 4, requires_grad=True)
        grl(x1).sum().backward()
        assert torch.allclose(x1.grad, torch.full_like(x1, -1.0))

        grl.lambda_ = 0.5
        x2 = torch.randn(2, 4, requires_grad=True)
        grl(x2).sum().backward()
        assert torch.allclose(x2.grad, torch.full_like(x2, -0.5))


class TestDomainAdversarialHead:
    def test_output_shape(self):
        head = DomainAdversarialHead(feat_dim=16, num_classes=4)
        feats = torch.randn(3, 16)
        logits = head(feats)
        assert logits.shape == (3, 4)

    def test_reverse_true_negates_upstream_gradient(self):
        head = DomainAdversarialHead(feat_dim=8, num_classes=2, reverse=True, lambda_=1.0)
        feats = torch.randn(4, 8, requires_grad=True)
        logits = head(feats)
        loss = logits.sum()
        loss.backward()

        # Reversal flips the sign of d(loss)/d(feats) relative to what a
        # plain (non-reversed) head with identical weights would produce.
        plain_head = DomainAdversarialHead(
            feat_dim=8, num_classes=2, reverse=False
        )
        plain_head.classifier.load_state_dict(head.classifier.state_dict())
        feats2 = feats.detach().clone().requires_grad_(True)
        plain_head(feats2).sum().backward()

        assert torch.allclose(feats.grad, -feats2.grad, atol=1e-6)

    def test_reverse_false_is_plain_multitask_head(self):
        head = DomainAdversarialHead(feat_dim=8, num_classes=3, reverse=False)
        feats = torch.randn(2, 8, requires_grad=True)
        head(feats).sum().backward()  # must run without error

        # The gradient reaches the features unreversed, whatever lambda_ is.
        head2 = DomainAdversarialHead(
            feat_dim=8, num_classes=3, reverse=False, lambda_=5.0
        )
        head2.classifier.load_state_dict(head.classifier.state_dict())
        feats2 = feats.detach().clone().requires_grad_(True)
        head2(feats2).sum().backward()
        assert torch.allclose(feats.grad, feats2.grad)
        reversed_head = DomainAdversarialHead(feat_dim=8, num_classes=3, reverse=True)
        reversed_head.classifier.load_state_dict(head.classifier.state_dict())
        feats3 = feats.detach().clone().requires_grad_(True)
        reversed_head(feats3).sum().backward()
        assert torch.allclose(feats.grad, -feats3.grad, atol=1e-6)

    def test_gradient_scales_with_lambda(self):
        head = DomainAdversarialHead(feat_dim=6, num_classes=2, reverse=True, lambda_=2.0)
        feats = torch.randn(3, 6, requires_grad=True)
        head(feats).sum().backward()
        grad_at_2 = feats.grad.clone()

        head2 = DomainAdversarialHead(feat_dim=6, num_classes=2, reverse=True, lambda_=4.0)
        head2.classifier.load_state_dict(head.classifier.state_dict())
        feats2 = feats.detach().clone().requires_grad_(True)
        head2(feats2).sum().backward()

        assert torch.allclose(feats2.grad, 2.0 * grad_at_2, atol=1e-5)


class _Util:
    def debug(self, message):
        pass

    def error(self, message):
        raise RuntimeError(message)


def _df():
    return pd.DataFrame(
        {"source_db": ["a", "a", "b", "c"], "language": ["en", "ja", "en", "ja"]}
    )


def _cfg(columns, reverse=True):
    return DannConfig(columns=columns, lambda_=1.0, weight=1.0, reverse=reverse)


class TestDannHeads:
    def test_off_returns_none(self):
        assert DannHeads.build(_df(), 8, _cfg([]), _Util()) is None

    def test_one_head_per_column_sized_by_distinct_values(self):
        heads = DannHeads.build(_df(), 8, _cfg(["source_db", "language"]), _Util())
        assert heads.heads["source_db"].classifier[-1].out_features == 3
        assert heads.heads["language"].classifier[-1].out_features == 2

    def test_encode_maps_values_to_sorted_indices(self):
        heads = DannHeads.build(_df(), 8, _cfg(["source_db", "language"]), _Util())
        encoded = heads.encode(_df())
        assert encoded.shape == (4, 2)
        assert encoded[:, 0].tolist() == [0, 0, 1, 2]
        assert encoded[:, 1].tolist() == [0, 1, 0, 1]

    def test_mixed_type_column_values_do_not_break_ordering(self):
        # e.g. a numeric column in which a missing value was filled with "na"
        df = pd.DataFrame({"age_group": [1, 2, "na", 1]})
        heads = DannHeads.build(df, 8, _cfg(["age_group"]), _Util())
        assert heads.encode(df).shape == (4, 1)
        assert len(heads.label_maps["age_group"]) == 3

    def test_missing_column_is_an_error(self):
        with pytest.raises(RuntimeError, match="not a column"):
            DannHeads.build(_df(), 8, _cfg(["nope"]), _Util())

    def test_single_valued_column_is_an_error(self):
        df = _df().assign(source_db="a")
        with pytest.raises(RuntimeError, match="at least 2"):
            DannHeads.build(df, 8, _cfg(["source_db"]), _Util())

    def test_missing_values_are_an_error(self):
        df = _df()
        df.loc[0, "source_db"] = None
        with pytest.raises(RuntimeError, match="missing values"):
            DannHeads.build(df, 8, _cfg(["source_db"]), _Util())

    def test_loss_sums_one_term_per_column(self):
        torch.manual_seed(0)
        heads = DannHeads.build(_df(), 8, _cfg(["source_db", "language"]), _Util())
        feats = torch.randn(4, 8)
        dom = torch.as_tensor(heads.encode(_df()))
        assert heads.loss(feats, dom).item() > 0

    def test_reversal_flips_feature_gradient_sign(self):
        grads = {}
        for reverse in (True, False):
            torch.manual_seed(0)
            heads = DannHeads.build(_df(), 8, _cfg(["source_db"], reverse), _Util())
            feats = torch.randn(4, 8, requires_grad=True)
            dom = torch.as_tensor(heads.encode(_df()))
            heads.loss(feats, dom).backward()
            grads[reverse] = feats.grad.clone()
        assert torch.allclose(grads[True], -grads[False], atol=1e-6)


def test_build_puts_heads_on_the_requested_device():
    """Regression: heads were left on CPU while features were on CUDA."""
    heads = DannHeads.build(_df(), 8, _cfg(["source_db"]), _Util(), device="meta")
    assert all(p.device.type == "meta" for p in heads.parameters())
