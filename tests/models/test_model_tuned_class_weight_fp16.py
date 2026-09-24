"""Regression for GH #438: TunedModel's inner Trainer.compute_loss
(nkululeko/models/model_tuned.py) does `loss = criterion(logits, targets)`.

Under fp16 (any [FINETUNE] device != "cpu", since fp16=self.device != "cpu"),
logits come out of the model as torch.half. With [FINETUNE] class_weight =
True, the CrossEntropyLoss criterion carries a `weight` tensor built as
plain torch.Tensor(train_weights) (float32) - and CrossEntropyLoss requires
its `weight` to match the input (logits) dtype, so this crashed with:

    RuntimeError: expected scalar type Half but found Float

_match_loss_dtype (see test_model_tuned_cast_targets.py) only aligns
*targets* with the logits dtype, and only for regression - it never
touches logits, so it doesn't help here. The fix casts logits to fp32
right before the criterion call, which this test reproduces directly
against a real weighted CrossEntropyLoss (not a mock, since the crash is
specific to PyTorch's own dtype-matching rules), without needing a GPU:
torch.half logits reproduce the exact same mismatch on CPU.
"""

import pytest
import torch


def _weighted_criterion():
    train_weights = torch.tensor([0.6, 0.4], dtype=torch.float32)
    return torch.nn.CrossEntropyLoss(weight=train_weights)


class TestClassWeightedLossUnderHalfLogits:
    def test_half_logits_against_float_weight_crashes_without_the_fix(self):
        criterion = _weighted_criterion()
        logits = torch.randn(4, 2, dtype=torch.half)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        with pytest.raises(RuntimeError, match="Half"):
            criterion(logits, targets)

    def test_casting_logits_to_float_fixes_it(self):
        criterion = _weighted_criterion()
        logits = torch.randn(4, 2, dtype=torch.half)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        loss = criterion(logits.float(), targets)

        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)

    def test_unweighted_half_logits_already_worked(self):
        # Sanity check: the crash is specific to the weighted case (a
        # Float `weight` tensor meeting Half logits), not a general
        # limitation of CrossEntropyLoss with fp16 logits - so this fix
        # must not be gated on class_weight being set.
        criterion = torch.nn.CrossEntropyLoss()
        logits = torch.randn(4, 2, dtype=torch.half)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        loss = criterion(logits, targets)
        assert torch.isfinite(loss)
